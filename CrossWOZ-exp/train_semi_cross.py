from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers import GPT2LMHeadModel
from transformers import BertTokenizer
from eval_cross import CrossWOZEvaluator
from reader_cross import CrossWozReader
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import shutil
import random
import argparse
import time
import logging
import json, copy, math
import utils
import numpy as np
from compute_joint_acc import compute_jacc
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from config_cross import global_config as cfg


import warnings
warnings.filterwarnings("ignore")

class Modal(object):
    
    def __init__(self, device=[0]):
        if len(device)>1:
            self.device1=device[0]
            self.device2=device[1]
        else:
            self.device1 = device[0]
            self.device2=device[0]
        if cfg.mode=='semi_VL':
            logging.info('PrioriModel sets on GPU{}, PosteriorModel sets on GPU{}'.format(self.device1,self.device2))
            tokenizer_path=cfg.PrioriModel_path
        else:
            tokenizer_path=cfg.gpt_path
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        # cfg.tokenizer = tokenizer

        # initialize multiwoz reader
        logging.info('hotel database path:{}'.format(cfg.dbs['hotel']))
        self.reader = CrossWozReader(self.tokenizer)
        self.sos_b_id=self.tokenizer.convert_tokens_to_ids('<sos_b>')
        self.sos_a_id=self.tokenizer.convert_tokens_to_ids('<sos_a>')
        self.sos_r_id=self.tokenizer.convert_tokens_to_ids('<sos_r>')
        self.eos_b_id=self.tokenizer.convert_tokens_to_ids('<eos_b>')
        self.eos_a_id=self.tokenizer.convert_tokens_to_ids('<eos_a>')
        self.eos_r_id=self.tokenizer.convert_tokens_to_ids('<eos_r>')
        self.sos_db_id=self.tokenizer.convert_tokens_to_ids('<sos_db>')
        self.eos_db_id=self.tokenizer.convert_tokens_to_ids('<eos_db>')
        logging.info([self.sos_b_id, self.sos_a_id, self.sos_r_id, self.eos_b_id, self.eos_a_id,self.eos_r_id])

        # create model: gpt2
        single_mode=['pretrain','train','semi_ST','test_pos']
        if cfg.mode in single_mode:
            self.model=GPT2LMHeadModel.from_pretrained(cfg.gpt_path)
            self.model.resize_token_embeddings(len(self.tokenizer))
            if cfg.gradient_checkpoint:
                self.model.config.gradient_checkpointing=True
            
            self.model.to(self.device1)
            self.PrioriModel=self.model
            if cfg.posterior_train:
                logging.info("Posterior model loaded from {}".format(cfg.gpt_path))
            else:
                logging.info("Prior model loaded from {}".format(cfg.gpt_path))
        
        elif cfg.mode=='test' or cfg.mode=='test_all':
            self.PrioriModel=GPT2LMHeadModel.from_pretrained(cfg.gpt_path)
            self.model=self.PrioriModel
            if cfg.gradient_checkpoint:
                self.PrioriModel.config.gradient_checkpointing=True
            self.PosteriorModel=None
            self.PrioriModel.to(self.device1)
        
        elif cfg.mode in ['semi_VL', 'semi_jsa']:#semi-VL
            self.PrioriModel=GPT2LMHeadModel.from_pretrained(cfg.PrioriModel_path)
            self.PosteriorModel=GPT2LMHeadModel.from_pretrained(cfg.PosteriorModel_path)
            logging.info("model loaded from {} and {}".format(cfg.PrioriModel_path,cfg.PosteriorModel_path))
            self.PrioriModel.resize_token_embeddings(len(self.tokenizer))
            self.PosteriorModel.resize_token_embeddings(len(self.tokenizer))
            if cfg.gradient_checkpoint:
                self.PrioriModel.config.gradient_checkpointing=True
                self.PosteriorModel.config.gradient_checkpointing=True
            self.PrioriModel.to(self.device1)
            self.PosteriorModel.to(self.device2)

        self.vocab_size=len(self.tokenizer)
        #
        self.evaluator = CrossWOZEvaluator(self.reader)
        if cfg.save_log:
            log_path='./log21/log_{}'.format(cfg.exp_no) if cfg.dataset==1 else './log/log_{}'.format(cfg.exp_no)
            if os.path.exists(log_path):
                shutil.rmtree(log_path)
                os.mkdir(log_path)
            else:
                os.mkdir(log_path)
            self.tb_writer = SummaryWriter(log_dir=log_path)
        else:
            self.tb_writer = None
        cfg.origin_batch_size=cfg.batch_size

        self.nll_loss=nn.NLLLoss(ignore_index=cfg.pad_id)
        self.eps=1e-45
        if 'test' not in cfg.mode:
            json.dump(cfg.__dict__,open(os.path.join(cfg.exp_path,'cfg_all.json'),'w'),indent=2)
        self.global_output=4
        #cfg._init_logging_handler(cfg.mode)

    def pretrain_turn_level(self, posterior=False):
        if cfg.mode=='train':
            num_dials=len(self.reader.train)
            all_batches = self.reader.get_batches('train')
        else:
            #divide the datasets
            if not os.path.exists(cfg.divided_path):
                train_data=self.reader.train
                temp_path=os.path.join(cfg.data_path,'divided_data{}.json'.format(cfg.spv_proportion-5))
                if os.path.exists(temp_path):
                    encoded_data = json.loads(open(temp_path, 'r', encoding='utf-8').read())
                    add_len=int(0.05*len(train_data))
                    self.pre_data=encoded_data['pre_data']+encoded_data['post_data'][:add_len]
                    self.post_data=encoded_data['post_data'][add_len:]
                    encoded_data={'pre_data':self.pre_data,'post_data':self.post_data}
                    logging.info('Divide data from %s, saved in %s'%(temp_path, cfg.divided_path))
                    json.dump(encoded_data, open(cfg.divided_path, 'w'), indent=2)
                else:
                    random.shuffle(train_data)
                    bound=int(len(train_data)*int(cfg.spv_proportion)/100)
                    self.pre_data=train_data[:bound]
                    self.post_data=train_data[bound:]
                    encoded_data={'pre_data':self.pre_data,'post_data':self.post_data}
                    logging.info('Divided data saved in %s'%cfg.divided_path)
                    json.dump(encoded_data, open(cfg.divided_path, 'w'), indent=2)
            else:
                encoded_data = json.loads(open(cfg.divided_path, 'r', encoding='utf-8').read())
                self.pre_data=encoded_data['pre_data']
                num_dials=len(self.pre_data)
            all_batches = self.reader.get_batches('train',data=self.pre_data)
            num_dials=len(self.pre_data)
        set_stats = self.reader.set_stats['train']
        num_turns=set_stats['num_turns']
        optimizer, scheduler = self.get_sep_optimizers(num_turns,self.model)

        # log info
        logging.info("***** Running turn-level training *****")
        logging.info("  Num Training steps(one turn in a batch of dialogs) per epoch = %d",
                     set_stats['num_training_steps_per_epoch'])
        logging.info("  Num Turns = %d", set_stats['num_turns'])
        logging.info("  Num Dialogs = %d", set_stats['num_dials'])
        logging.info("  Num Epochs = %d", cfg.epoch_num)
        logging.info("  Batch size  = %d", cfg.batch_size)
        logging.info("  Gradient Accumulation steps = %d",
                     cfg.gradient_accumulation_steps)
        logging.info("  Total optimization steps = %d",
                     set_stats['num_training_steps_per_epoch']*cfg.epoch_num // cfg.gradient_accumulation_steps)

        log_inputs = 4
        global_step = 0

        min_loss = 1000
        min_eval_loss=1000
        max_score=0
        early_stop_count=cfg.early_stop_count
        epoch_th=0.2*cfg.epoch_num if 'gpt2' in cfg.gpt_path else -1
        warmup_epochs=cfg.warmup_steps*cfg.gradient_accumulation_steps*cfg.batch_size//num_dials \
            if cfg.warmup_steps>=0 else int(cfg.epoch_num*cfg.warmup_ratio)
        c1, c2=0,0
        for epoch in range(cfg.epoch_num):
            epoch_step = 0
            tr_loss = 0.0
            btm = time.time()
            oom_time = 0
            self.model.zero_grad()
            random.shuffle(all_batches)
            #data_iterator = self.reader.get_data_iterator(all_batches)

            for batch_idx, batch0 in enumerate(all_batches):
                dial_batch=self.reader.transpose_batch(batch0)
                pv_batch = None
                c1+=1
                for turn_num, turn_batch in enumerate(dial_batch):
                    c2+=1
                    first_turn = (turn_num == 0)
                    inputs, labels = self.reader.convert_batch_turn(turn_batch, pv_batch, first_turn, posterior=posterior)
                    pv_batch = self.reader.get_pv_batch(pv_batch, turn_batch['user'],
                        turn_batch['resp'], turn_batch['bspn'])
                    try:  # avoid OOM
                        self.model.train()
                        if log_inputs > 0:  # log inputs for the very first two turns
                            logging.info('Input examples:\n{}'.format(self.tokenizer.decode(inputs['contexts'][0])))
                            log_inputs-=1
                        inputs = self.add_torch_input(inputs)
                        outputs = self.model(inputs['contexts_tensor'])
                        if cfg.only_target_loss:
                            labels=self.add_torch_input(labels)    
                            loss = self.calculate_loss_and_accuracy(outputs, labels=labels['contexts_tensor'])
                        else:
                            loss = self.calculate_loss_and_accuracy(outputs, labels=inputs['contexts_tensor'])
                        loss.backward()
                        tr_loss += loss.item()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                        epoch_step += 1

                        # step, wrt gradient_accumulation_steps, clip grad norm
                        if epoch_step % cfg.gradient_accumulation_steps == 0 or(
                            batch_idx==len(all_batches)-1 and turn_num==len(dial_batch)-1):
                            optimizer.step()
                            scheduler.step()
                            optimizer.zero_grad()
                            # global_step: actual step the optimizer took
                            global_step += 1
                            

                    except RuntimeError as exception:
                        if "out of memory" in str(exception):
                            max_length = max(inputs['lengths'])
                            oom_time += 1
                            logging.info("WARNING: ran out of memory,times: {}, batch size: {}, max_len: {}".format(
                                oom_time, cfg.batch_size, max_length))
                            if hasattr(torch.cuda, 'empty_cache'):
                                torch.cuda.empty_cache()
                        else:
                            logging.info(str(exception))
                            raise exception
            if epoch==0:
                logging.info('Num dials:{}, num_turns:{}'.format(c1, c2))
            logging.info('Epoch:{}, Train epoch time: {:.2f} min, epoch loss: {:.4f}'.format(
                epoch, (time.time()-btm)/60, tr_loss))
            if cfg.evaluate_during_training:
                if cfg.save_type=='min_loss':
                    eval_loss=self.eval(model=self.model,posterior=posterior)
                    logging.info('model evaluation loss:{}'.format(eval_loss))
                    if self.tb_writer:
                        self.tb_writer.add_scalar('loss_eval',eval_loss,epoch)
                    if eval_loss<min_eval_loss:
                        min_eval_loss=eval_loss
                        self.save_model(path='best_loss_model',model=self.model)
                        early_stop_count=cfg.early_stop_count
                    else:
                        if epoch>=warmup_epochs:#early stop after warm up
                            early_stop_count-=1
                            logging.info('early stop count:%d'%early_stop_count)
                            if early_stop_count==0 and cfg.early_stop:
                                logging.info('early stopped')
                                break
                elif cfg.save_type=='max_score' and epoch>epoch_th:
                    if posterior:
                        eval_result=self.validate_pos(data='dev')
                        self.tb_writer.add_scalar('joint_goal',eval_result['joint_acc'],epoch)
                        self.tb_writer.add_scalar('act_F1',eval_result['act_F1'],epoch)
                        self.tb_writer.add_scalar('db_acc',eval_result['db_acc'],epoch)
                        score=eval_result['joint_acc']
                    else:
                        eval_result=self.validate_fast(data='dev')
                        self.tb_writer.add_scalar('joint_goal',eval_result['joint_acc'],epoch)
                        self.tb_writer.add_scalar('match',eval_result['match'],epoch)
                        self.tb_writer.add_scalar('success',eval_result['success'],epoch)
                        self.tb_writer.add_scalar('bleu',eval_result['bleu'],epoch)
                        self.tb_writer.add_scalar('combined_score',eval_result['score'],epoch)
                        score=eval_result['score']
                    if score>max_score:
                        early_stop_count=cfg.early_stop_count
                        max_score=score
                        self.save_model(path='best_score_model',model=self.model)
                    else:
                        if epoch>=warmup_epochs:
                            early_stop_count-=1
                            logging.info('early stop count:%d'%early_stop_count)
                            if early_stop_count==0 and cfg.early_stop:
                                logging.info('early stopped')
                                break
            else:#save the model with minimal training loss
                pass
    
    def semi_ST(self):
        logging.info('------Running self training------')
        data=json.loads(open(cfg.divided_path, 'r', encoding='utf-8').read())
        data_lab=data['pre_data']
        data_unl=data['post_data']
            
        logging.info('Labeled dials:{}, unlabeled dials:{}'.format(len(data_lab),len(data_unl)))
        num_dials=len(data_lab)+len(data_unl)
        
        cfg.batch_size=cfg.batch_size*cfg.gradient_accumulation_steps
        batches_lab=self.reader.get_batches('train',data=data_lab)
        batches_unl=self.reader.get_batches('train',data=data_unl)
        all_batches=[]
        data_repeate=30//cfg.spv_proportion if cfg.spv_proportion<=15 else 1
        for _ in range(data_repeate-1):
            num_dials+=len(data_lab)

        if cfg.debugging:
            batches_unl=[]
            #batches_unl=batches_unl[-len(batches_unl)//10:]
            #batches_lab=[]
        
        for _ in range(data_repeate):
            for batch in batches_lab:
                all_batches.append({'batch':self.reader.transpose_batch(batch),'supervised':True})
        
        for batch in batches_unl:
            all_batches.append({'batch':self.reader.transpose_batch(batch),'supervised':False})
        batch_num=sum([len(item['batch']) for item in all_batches])
        logging.info('Total steps:{}'.format(batch_num))
        # cleare memory
        batches_lab=[]
        batches_unl=[]
        optimizer, scheduler = self.get_sep_optimizers(num_dials, self.model, num_batches=batch_num)

        # log info
        logging.info("  Num Epochs = %d", cfg.epoch_num)
        logging.info("  Batch size  = %d", cfg.batch_size)
        logging.info('  Num Batches = %d', len(all_batches))
        log_inputs = 3
        global_step = 0
        max_score=0
        if cfg.init_eval:
            self.validate_fast(data='dev')
        early_stop_count=cfg.early_stop_count
        if cfg.use_scheduler:
            warmup_epochs=cfg.warmup_steps*cfg.batch_size//num_dials if cfg.warmup_steps>=0 else int(cfg.epoch_num*cfg.warmup_ratio)
            logging.info('Warmup epochs:{}'.format(warmup_epochs))
        weight_decay_count=cfg.weight_decay_count
        lr=cfg.lr
        for epoch in range(cfg.epoch_num):
            epoch_step = 0
            tr_loss, sup_loss, uns_loss = 0.0, 0.0, 0.0
            sup_step, uns_step=0, 0
            btm = time.time()
            self.model.zero_grad()
            random.shuffle(all_batches)

            for batch_idx, dial_batch_dict in enumerate(all_batches):
                pv_batch=None
                pv_bspn_batch=None
                turn_domain_batch=[[] for _ in range(len(dial_batch_dict['batch'][0]['dial_id']))]
                for turn_num, turn_batch in enumerate(dial_batch_dict['batch']):
                    if dial_batch_dict['supervised']==False:
                        turn_batch, next_pv_batch, pv_bspn_batch, turn_domain_batch=self.gen_hidden_state(turn_batch,
                            pv_batch, posterior=False, pv_bspn_batch=pv_bspn_batch, turn_domain_batch=turn_domain_batch)
                    else:
                        next_pv_batch=self.reader.get_pv_batch(pv_batch, user=turn_batch['user'], 
                            resp=turn_batch['resp'], bspn=turn_batch['bspn'])
                    first_turn = (turn_num == 0)
                    mini_batches, mini_pv_batches=self.reader.split_turn_batch(turn_batch, cfg.origin_batch_size, other_batch=pv_batch)
                    for i, batch in enumerate(mini_batches):
                        mini_pv_batch=None if turn_num==0 else mini_pv_batches[i]
                        if not dial_batch_dict['supervised']:# unsupervised training
                            inputs, labels, seg_labels=self.reader.convert_batch_turn(batch, mini_pv_batch, first_turn, posterior=False, seg_label=True)
                            inputs = self.add_torch_input(inputs)
                            labels=self.add_torch_input(labels)
                            if log_inputs > 0 and cfg.example_log:  # log inputs for the very first two turns
                                logging.info('Examples\n{}'.format(self.tokenizer.decode(inputs['contexts'][0])))
                                log_inputs-=1
                            outputs = self.model(inputs['contexts_tensor'])
                            ST_inputs, resp_label=self.get_ST_input1(inputs['contexts_tensor'], outputs[0], labels['contexts_tensor'], list(seg_labels['contexts_np']))
                            embeds=ST_inputs.matmul(self.model.get_input_embeddings().weight)
                            outputs=self.model(inputs_embeds=embeds)
                            loss=self.calculate_loss_and_accuracy(outputs, resp_label)
                            if cfg.loss_reg:
                                loss=loss/cfg.gradient_accumulation_steps
                            loss.backward()
                            tr_loss+=loss.item()
                            uns_loss+=loss.item()
                            uns_step+=1
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                        else:
                            inputs, labels=self.reader.convert_batch_turn(batch, mini_pv_batch, first_turn, posterior=False)
                            inputs = self.add_torch_input(inputs)
                            outputs = self.model(inputs['contexts_tensor'])
                            labels=self.add_torch_input(labels)    
                            loss = self.calculate_loss_and_accuracy(outputs, labels=labels['contexts_tensor'])
                            if cfg.loss_reg:
                                loss=loss/cfg.gradient_accumulation_steps
                            loss.backward()
                            tr_loss += loss.item()
                            sup_loss+=loss.item()
                            sup_step+=1
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                    epoch_step+=1
                    global_step+=1
                    tr_loss+=loss.item()
                    optimizer.step()
                    optimizer.zero_grad()
                    if cfg.use_scheduler:
                        scheduler.step()
                    if self.tb_writer:
                        self.tb_writer.add_scalar('lr', optimizer.param_groups[0]["lr"],global_step)
                        self.tb_writer.add_scalar('loss', loss.item(), global_step)
                    pv_batch=next_pv_batch
            
            logging.info('Epoch: {}, Train epoch time: {:.2f} min, loss:{:.3f}, avg_sup_loss:{:.3f}, avg_uns_loss:{:.3f}'.format(epoch, 
                (time.time()-btm)/60, tr_loss/epoch_step, sup_loss/(sup_step+1e-10), uns_loss/(uns_step+1e-10)))

            eval_result=self.validate_fast(data='dev')
            if self.tb_writer:
                self.tb_writer.add_scalar('joint_goal',eval_result['joint_acc'],epoch)
                self.tb_writer.add_scalar('match',eval_result['match'],epoch)
                self.tb_writer.add_scalar('success',eval_result['success'],epoch)
                self.tb_writer.add_scalar('bleu',eval_result['bleu'],epoch)
                self.tb_writer.add_scalar('combined_score',eval_result['score'],epoch)
            
            if eval_result['score']>max_score:
                max_score=eval_result['score']
                self.save_model(path='best_score_model')
                early_stop_count=cfg.early_stop_count
            else:
                weight_decay_count-=1
                if weight_decay_count==0 and not cfg.use_scheduler:
                    lr=lr*cfg.lr_decay
                    for group in optimizer.param_groups:
                        group['lr'] = lr
                    logging.info("learning rate decay to {}".format(lr))
                    weight_decay_count = cfg.weight_decay_count
                if epoch>=warmup_epochs:
                    early_stop_count-=1
                    logging.info('early stop count:%d'%early_stop_count)
            if lr<1e-9 and not cfg.use_scheduler:
                logging.info('learning rate too small, break')
                break
            if early_stop_count==0 and cfg.early_stop:
                logging.info('early stopped')
                break


    def semi_VL(self):
        logging.info('------Running variational learning------')
        data=json.loads(open(cfg.divided_path, 'r', encoding='utf-8').read())
        data_lab=data['pre_data']
        data_unl=data['post_data']
        logging.info('Labeled dials:{}, unlabeled dials:{}'.format(len(data_lab),len(data_unl)))
        num_dials=len(data_lab)+len(data_unl)
        
        cfg.batch_size=cfg.batch_size*cfg.gradient_accumulation_steps
        batches_lab=self.reader.get_batches('train',data=data_lab)
        label_turns=self.reader.set_stats['train']['num_turns']
        batches_unl=self.reader.get_batches('train',data=data_unl)
        unlabel_turns=self.reader.set_stats['train']['num_turns']
        all_batches=[]
        data_repeate=3 if cfg.spv_proportion<=15 else 1
        label_turns*=data_repeate
        num_turns=label_turns+unlabel_turns
        for _ in range(data_repeate-1):
            num_dials+=len(data_lab)

        if cfg.debugging:
            batches_lab=[]
            #batches_unl=[]
            batches_unl=batches_unl[:len(batches_unl)//15]

        for _ in range(data_repeate):
            for batch in batches_lab:
                all_batches.append({'batch':self.reader.transpose_batch(batch),'supervised':True})
        for batch in batches_unl:
            all_batches.append({'batch':self.reader.transpose_batch(batch),'supervised':False})
        batch_num=sum([len(item['batch']) for item in all_batches])
        logging.info('Total turns:{}, steps:{}'.format(num_turns, batch_num))
        # cleare memory
        batches_lab=[]
        batches_unl=[]
        optimizer1, scheduler1 = self.get_sep_optimizers(num_turns,self.PrioriModel, num_batches=batch_num)
        optimizer2, scheduler2 = self.get_sep_optimizers(num_turns,self.PosteriorModel, num_batches=batch_num)

        # log info
        logging.info("  Num Epochs = %d", cfg.epoch_num)
        logging.info("  Batch size  = %d", cfg.batch_size)
        logging.info('  Num Batches = %d', len(all_batches))
        log_inputs = 3
        global_step = 0
        max_score=0
        early_stop_count=cfg.early_stop_count
        if cfg.use_scheduler:
            warmup_epochs=cfg.warmup_steps*cfg.batch_size//num_dials if cfg.warmup_steps>=0 else int(cfg.epoch_num*cfg.warmup_ratio)
            logging.info('Warmup epochs:{}'.format(warmup_epochs))
        weight_decay_count=cfg.weight_decay_count
        lr=cfg.lr
        for epoch in range(cfg.epoch_num):
            epoch_step = 0
            tr_loss, sup_loss, uns_loss = 0.0, 0.0, 0.0
            sup_step, uns_step=0, 0
            btm = time.time()
            self.PrioriModel.zero_grad()
            self.PosteriorModel.zero_grad()
            random.shuffle(all_batches)

            for batch_idx, dial_batch_dict in enumerate(all_batches):
                pv_batch=None
                pv_bspn_batch=None
                turn_domain_batch=[[] for _ in range(len(dial_batch_dict['batch'][0]['dial_id']))]
                for turn_num, turn_batch in enumerate(dial_batch_dict['batch']):
                    if dial_batch_dict['supervised']==False:
                        turn_batch, next_pv_batch, pv_bspn_batch, turn_domain_batch=self.gen_hidden_state(turn_batch,
                            pv_batch, posterior=True, pv_bspn_batch=pv_bspn_batch, turn_domain_batch=turn_domain_batch)
                    else:
                        next_pv_batch=self.reader.get_pv_batch(pv_batch, user=turn_batch['user'], 
                            resp=turn_batch['resp'], bspn=turn_batch['bspn'])
                    first_turn = (turn_num == 0)
                    mini_batches, mini_pv_batches=self.reader.split_turn_batch(turn_batch, cfg.origin_batch_size, other_batch=pv_batch)
                    for i, batch in enumerate(mini_batches):
                        mini_pv_batch=None if turn_num==0 else mini_pv_batches[i]
                        if not dial_batch_dict['supervised']:# unsupervised training
                            inputs_prior, labels_prior = self.reader.convert_batch_turn(batch, mini_pv_batch, first_turn, posterior=False)
                            inputs_posterior, labels_posterior = self.reader.convert_batch_turn(batch, mini_pv_batch, first_turn, posterior=True)
                            self.PrioriModel.train()
                            self.PosteriorModel.train()
                            if log_inputs > 0 and cfg.example_log:  # log inputs for the very first two turns
                                logging.info('Prior examples:\n{}'.format(self.tokenizer.decode(inputs_prior['contexts'][0])))
                                logging.info("Posterior examples:\n{}".format(self.tokenizer.decode(inputs_posterior['contexts'][0])))
                                log_inputs -= 1

                            # to tensor
                            inputs_prior = self.add_torch_input(inputs_prior)#B,T
                            inputs_posterior = self.add_torch_input(inputs_posterior,posterior=True)
                            labels_prior=self.add_torch_input(labels_prior)#B,T
                            labels_posterior=self.add_torch_input(labels_posterior,posterior=True)
                            # loss
                            outputs_prior=self.PrioriModel(inputs_prior['contexts_tensor'])
                            outputs_posterior=self.PosteriorModel(inputs_posterior['contexts_tensor'])#B,T,V
                            logits_pri=outputs_prior[0]
                            logits_post=outputs_posterior[0]
                            #straight through trick
                            ST_inputs_prior, resp_label=self.get_ST_input(inputs_prior['contexts_tensor'],\
                                    logits_post,labels_prior['contexts_tensor'],labels_posterior['contexts_tensor'])
                            loss_kl=self.get_kl_loss(logits_pri,logits_post.to(self.device1),\
                                    labels_prior['contexts_tensor'],labels_posterior['contexts_tensor'].to(self.device1))
                            
                            embed_prior=ST_inputs_prior.matmul(self.PrioriModel.get_input_embeddings().weight)#multiple the input embedding
                            outputs1=self.PrioriModel(inputs_embeds=embed_prior)
                            loss_ce=self.calculate_loss_and_accuracy(outputs1, resp_label)
                            if torch.isnan(loss_kl) or torch.isnan(loss_ce):
                                t=1
                            loss=loss_ce+cfg.kl_loss_weight*loss_kl
                            if cfg.loss_reg:
                                loss=loss/cfg.gradient_accumulation_steps
                            loss.backward()
                            tr_loss+=loss.item()
                            uns_loss+=loss.item()
                            uns_step+=1
                            torch.nn.utils.clip_grad_norm_(self.PrioriModel.parameters(), 5.0)
                            torch.nn.utils.clip_grad_norm_(self.PosteriorModel.parameters(), 5.0)
                        else:# supervised training
                            inputs_prior, labels_prior = self.reader.convert_batch_turn(batch, mini_pv_batch, first_turn, posterior=False)
                            inputs_posterior, labels_posterior = self.reader.convert_batch_turn(batch, mini_pv_batch, first_turn, posterior=True)
                            inputs_prior = self.add_torch_input(inputs_prior)#B,T
                            labels_prior=self.add_torch_input(labels_prior)#B,T
                            inputs_posterior=self.add_torch_input(inputs_posterior,posterior=True)
                            labels_posterior=self.add_torch_input(labels_posterior,posterior=True)

                            outputs1 = self.PrioriModel(inputs_prior['contexts_tensor'])
                            loss_pri=self.calculate_loss_and_accuracy(outputs1,labels_prior['contexts_tensor'])
                            outputs2=self.PosteriorModel(inputs_posterior['contexts_tensor'])
                            loss_pos=self.calculate_loss_and_accuracy(outputs2,labels_posterior['contexts_tensor'])

                            if cfg.loss_reg:
                                loss_pri=loss_pri/cfg.gradient_accumulation_steps
                                loss_pos=loss_pos/cfg.gradient_accumulation_steps
                            loss=loss_pri+loss_pos.to(self.device1)
                            loss.backward()
                            tr_loss+=loss.item()
                            sup_loss+=loss.item()
                            sup_step+=1
                            torch.nn.utils.clip_grad_norm_(self.PrioriModel.parameters(), 5.0)
                            torch.nn.utils.clip_grad_norm_(self.PosteriorModel.parameters(), 5.0)
                    epoch_step+=1
                    optimizer1.step()
                    optimizer1.zero_grad()
                    optimizer2.step()
                    optimizer2.zero_grad()
                    global_step+=1
                    if cfg.use_scheduler:
                        scheduler1.step()
                        scheduler2.step()
                    if self.tb_writer:
                        self.tb_writer.add_scalar('lr1', optimizer1.param_groups[0]["lr"],global_step)
                        self.tb_writer.add_scalar('lr2', optimizer2.param_groups[0]["lr"],global_step)
                        self.tb_writer.add_scalar('loss', loss.item(), global_step)
                    pv_batch=next_pv_batch
            
            logging.info('Epoch: {}, Train epoch time: {:.2f} min, loss:{:.3f}, avg_sup_loss:{:.3f}, avg_uns_loss:{:.3f}'.format(epoch, 
                (time.time()-btm)/60, tr_loss/epoch_step, sup_loss/(sup_step+1e-10), uns_loss/(uns_step+1e-10)))
            eval_result=self.validate_fast(data='dev')
            if self.tb_writer:
                self.tb_writer.add_scalar('joint_goal',eval_result['joint_acc'],epoch)
                self.tb_writer.add_scalar('match',eval_result['match'],epoch)
                self.tb_writer.add_scalar('success',eval_result['success'],epoch)
                self.tb_writer.add_scalar('bleu',eval_result['bleu'],epoch)
                self.tb_writer.add_scalar('combined_score',eval_result['score'],epoch)
            
            if eval_result['score']>max_score:
                max_score=eval_result['score']
                self.save_model(path='best_score_model')
                early_stop_count=cfg.early_stop_count
            else:
                weight_decay_count-=1
                if weight_decay_count==0 and not cfg.use_scheduler:
                    lr=lr*cfg.lr_decay
                    for group in optimizer1.param_groups:
                        group['lr'] = lr
                    for group in optimizer2.param_groups:
                        group['lr'] = lr
                    logging.info("learning rate decay to {}".format(lr))
                    weight_decay_count = cfg.weight_decay_count
                if epoch>=warmup_epochs:
                    early_stop_count-=1
                    logging.info('early stop count:%d'%early_stop_count)
            if lr<1e-9 and not cfg.use_scheduler:
                logging.info('learning rate too small, break')
                break
            if early_stop_count==0 and cfg.early_stop:
                logging.info('early stopped')
                break

    def gen_hidden_state(self, turn_batch, pv_batch, turn_num, posterior=True):
        if posterior:
            self.model=self.PosteriorModel
        else:
            self.model=self.PrioriModel
        self.model.eval()
        max_len_b=60
        max_len_a=20
        with torch.no_grad():
            # generate bspn
            contexts=self.reader.convert_eval_batch_turn(turn_batch,pv_batch, mode='gen_bspn', posterior=posterior)
            bspn_batch=self.generate_batch(self.model, contexts, max_len_b, self.eos_b_id)
            bs_gen, db_gen=self.get_bspn(bspn_batch, return_db=True, turn_domain=turn_batch['turn_domain'])
            # generate aspn
            contexts=self.reader.convert_eval_batch_turn(turn_batch,pv_batch, mode='gen_ar', 
                bspn_gen=bs_gen,db_gen=db_gen, posterior=posterior)
            aspn_batch=self.generate_batch(self.model, contexts, max_len_a, self.eos_a_id)
            aspn_gen=self.get_aspn(aspn_batch)
            turn_batch['bspn']=bs_gen
            turn_batch['db']=db_gen
            turn_batch['aspn']=aspn_gen
            pv_batch=self.reader.get_pv_batch(pv_batch, turn_batch['user'], turn_batch['resp'], bs_gen)
        return turn_batch, pv_batch

    def gen_hidden_state1(self, turn_batch, pv_batch, posterior=True, pv_bspn_batch=None, turn_domain_batch=None):
        if posterior:
            self.model=self.PosteriorModel
        else:
            self.model=self.PrioriModel
        self.model.eval()
        max_len_b=60
        max_len_a=20
        with torch.no_grad():
            # generate bspn
            contexts=self.reader.convert_eval_batch_turn(turn_batch,pv_batch, mode='gen_bspn', posterior=posterior)
            bspn_batch=self.generate_batch(self.model, contexts, max_len_b, self.eos_b_id)
            if cfg.use_true_domain_for_ctr_train:
                bs_gen, db_gen=self.get_bspn(bspn_batch, return_db=True, turn_domain=turn_batch['turn_domain'])
            else:
                bs_gen=self.get_bspn(bspn_batch)
                turn_domain_batch, db_gen=self.get_turn_domain(turn_domain_batch, bs_gen, pv_bspn_batch)
            # generate aspn
            contexts=self.reader.convert_eval_batch_turn(turn_batch,pv_batch, mode='gen_ar', 
                bspn_gen=bs_gen,db_gen=db_gen, posterior=posterior)
            aspn_batch=self.generate_batch(self.model, contexts, max_len_a, self.eos_a_id)
            aspn_gen=self.get_aspn(aspn_batch)
            turn_batch['bspn']=bs_gen
            turn_batch['db']=db_gen
            turn_batch['aspn']=aspn_gen
            pv_batch=self.reader.get_pv_batch(pv_batch, turn_batch['user'], turn_batch['resp'], bs_gen)
        return turn_batch, pv_batch, bs_gen, turn_domain_batch
        

    def get_ST_input(self,inputs, logits, labels1, labels2):
        #add straight through for variational learning
        #inputs:B,T1
        #logits:B,T1,V or B,T2,V
        #labels1:B,T1
        #labels2:B,T1 or B,T2
        onehot=F.one_hot(inputs,self.vocab_size).float() # B, T, V
        resp_label=cfg.pad_id*torch.ones(labels1.shape).long().to(labels1.device)
        for dial_idx in range(logits.size(0)):
            label_pri=labels1[dial_idx,:].ne(cfg.pad_id).long().cpu().tolist() #0 for pad token and 1 for hidden states tokens
            label_post=labels2[dial_idx,:].ne(cfg.pad_id).long().cpu().tolist()
            st_idx=label_post.index(1)
            h_len=len(label_post)-label_post[::-1].index(1)-st_idx
            probs=F.softmax(logits[dial_idx, st_idx:st_idx+h_len-1,:], dim=-1) # probs of hidden states
            st_idx=label_pri.index(1)
            onehot[dial_idx, st_idx+1:st_idx+h_len, :]+=(probs-probs.detach()).to(onehot.device)
            resp_label[dial_idx, st_idx+h_len:]=labels1[dial_idx, st_idx+h_len:]
        return onehot, resp_label
    
    def get_ST_input1(self, inputs, logits, labels, seg_labels):
        #add straight through for self training
        #inputs:B,T1
        #logits:B,T1,V or B,T2,V
        #labels1:B,T1
        #labels2:B,T1 or B,T2
        onehot=F.one_hot(inputs,self.vocab_size).float() # B, T, V
        resp_label=cfg.pad_id*torch.ones(labels.shape).long().to(labels.device)
        for dial_idx in range(logits.size(0)):
            sta_idx=list(seg_labels[dial_idx]).index(1)
            end_idx=list(seg_labels[dial_idx]).index(2)
            probs=F.softmax(logits[dial_idx, sta_idx:end_idx-1,:],dim=-1)
            onehot[dial_idx, sta_idx+1:end_idx, :]+=(probs-probs.detach()).to(onehot.device)
            resp_label[dial_idx, end_idx:]=labels[dial_idx, end_idx:]
        return onehot, resp_label

    def kl_loss(self, p_proba, q_proba): # [B, T, V] or [T,V]
        dim=p_proba.dim()
        loss = q_proba * (torch.log(q_proba+self.eps) - torch.log(p_proba+self.eps))
        loss = torch.sum(loss, dim=-1)   # sum over vocabulary
        loss = torch.sum(loss, dim=-1)   # sum over sequence
        if dim==2:
            return loss
        else:
            return loss.mean()

    def semi_jsa(self):
        logging.info('------Running joint stochastic approximation------')
        data=json.loads(open(cfg.divided_path, 'r', encoding='utf-8').read())
        data_lab=data['pre_data']
        data_unl=data['post_data']
        logging.info('Labeled dials:{}, unlabeled dials:{}'.format(len(data_lab),len(data_unl)))
        num_dials=len(data_lab)+len(data_unl)
        
        cfg.batch_size=cfg.batch_size*cfg.gradient_accumulation_steps
        batches_lab=self.reader.get_batches('train',data=data_lab)
        label_turns=self.reader.set_stats['train']['num_turns']
        batches_unl=self.reader.get_batches('train',data=data_unl)
        unlabel_turns=self.reader.set_stats['train']['num_turns']
        all_batches=[]
        data_repeate=1 if cfg.spv_proportion==10 else 1
        label_turns*=data_repeate
        num_turns=label_turns+unlabel_turns
        for _ in range(data_repeate-1):
            num_dials+=len(data_lab)

        if cfg.debugging:
            batches_lab=[]
            #batches_unl=[]
            batches_unl=batches_unl[:len(batches_unl)//15]

        for _ in range(data_repeate):
            for batch in batches_lab:
                all_batches.append({'batch':self.reader.transpose_batch(batch),'supervised':True})
        for batch in batches_unl:
            all_batches.append({'batch':self.reader.transpose_batch(batch),'supervised':False})
        batch_num=sum([len(item['batch']) for item in all_batches])
        logging.info('Total turns:{}, steps:{}'.format(num_turns, batch_num))
        # cleare memory
        batches_lab=[]
        batches_unl=[]
        optimizer1, scheduler1 = self.get_sep_optimizers(num_turns,self.PrioriModel, num_batches=batch_num)
        optimizer2, scheduler2 = self.get_sep_optimizers(num_turns,self.PosteriorModel, num_batches=batch_num)

        # log info
        logging.info("  Num Epochs = %d", cfg.epoch_num)
        logging.info("  Batch size  = %d", cfg.batch_size)
        logging.info('  Num Batches = %d', len(all_batches))
        log_inputs = 3
        global_step = 0
        max_score=0
        early_stop_count=cfg.early_stop_count
        if cfg.use_scheduler:
            warmup_epochs=cfg.warmup_steps*cfg.batch_size//num_dials if cfg.warmup_steps>=0 else int(cfg.epoch_num*cfg.warmup_ratio)
            logging.info('Warmup epochs:{}'.format(warmup_epochs))
        weight_decay_count=cfg.weight_decay_count
        lr=cfg.lr
        for epoch in range(cfg.epoch_num):
            epoch_step = 0
            tr_loss, sup_loss, uns_loss = 0.0, 0.0, 0.0
            sup_step, uns_step=0, 0
            btm = time.time()
            self.PrioriModel.zero_grad()
            self.PosteriorModel.zero_grad()
            random.shuffle(all_batches)

            for batch_idx, dial_batch_dict in enumerate(all_batches):
                pv_batch=None
                for turn_num, turn_batch in enumerate(dial_batch_dict['batch']):
                    if dial_batch_dict['supervised']==False:
                        turn_batch, next_pv_batch=self.gen_hidden_state(turn_batch, pv_batch, turn_num, posterior=True)
                    else:
                        next_pv_batch=self.reader.get_pv_batch(pv_batch, user=turn_batch['user'], 
                            resp=turn_batch['resp'], bspn=turn_batch['bspn'])
                    first_turn = (turn_num == 0)
                    mini_batches, mini_pv_batches=self.reader.split_turn_batch(turn_batch, cfg.origin_batch_size, other_batch=pv_batch)
                    for i, batch in enumerate(mini_batches):
                        mini_pv_batch=None if turn_num==0 else mini_pv_batches[i]
                        if not dial_batch_dict['supervised']:# unsupervised training
                            inputs_prior, labels_prior = self.reader.convert_batch_turn(batch, mini_pv_batch, first_turn, posterior=False)
                            inputs_posterior, labels_posterior = self.reader.convert_batch_turn(batch, mini_pv_batch, first_turn, posterior=True)
                            self.PrioriModel.train()
                            self.PosteriorModel.train()
                            if log_inputs > 0 and cfg.example_log:  # log inputs for the very first two turns
                                logging.info('Prior examples:\n{}'.format(self.tokenizer.decode(inputs_prior['contexts'][0])))
                                logging.info("Posterior examples:\n{}".format(self.tokenizer.decode(inputs_posterior['contexts'][0])))
                                log_inputs -= 1

                            jsa_labels=(copy.deepcopy(inputs_posterior),copy.deepcopy(labels_posterior),copy.deepcopy(inputs_prior),copy.deepcopy(labels_prior))
                            # to tensor
                            inputs_prior = self.add_torch_input(inputs_prior)#B,T
                            inputs_posterior = self.add_torch_input(inputs_posterior,posterior=True)
                            labels_prior=self.add_torch_input(labels_prior)#B,T
                            labels_posterior=self.add_torch_input(labels_posterior,posterior=True)
                            # loss
                            outputs_prior=self.PrioriModel(inputs_prior['contexts_tensor'])
                            outputs_posterior=self.PosteriorModel(inputs_posterior['contexts_tensor'])#B,T,V
                            logits_pri=outputs_prior[0]
                            logits_post=outputs_posterior[0]
                            
                            #get prob
                            jsa_prob=self.get_jsa_prob(logits_pri,logits_post,\
                                    labels_prior['contexts_tensor'],labels_posterior['contexts_tensor'])
                            if epoch==0:
                                last_prob=jsa_prob 
                                if 'prob' not in turn_batch:
                                    turn_batch['prob']=[]
                                turn_batch['prob'].append(jsa_prob)
                            else:
                                last_prob=copy.deepcopy(turn_batch['prob'][i])#accept the proposal at the first turn
                                turn_batch['prob'][i]=jsa_prob
                            
                            #update bspn
                            
                            for prob_num in range(min(len(jsa_prob),len(last_prob))):
                                if jsa_prob[prob_num]-last_prob[prob_num]>0:
                                    ratio=1.0
                                else:
                                    ratio=math.exp(jsa_prob[prob_num]-last_prob[prob_num])
                                if ratio<1.0:
                                    if random.random()>ratio:
                                        for j in range(4):
                                            if 'contexts_np' in jsa_labels[j]:
                                                jsa_labels[j].pop('contexts_np')
                                            jsa_labels[j]['contexts'][prob_num]=turn_batch['jsa_labels'][i][j]['contexts'][prob_num]
                                            #jsa_labels[j]['contexts_np'][prob_num]=dial_batch_dict['jsa_labels'][j]['contexts_np'][prob_num]
                                            jsa_labels[j]['lengths'][prob_num]=turn_batch['jsa_labels'][i][j]['lengths'][prob_num]                        
                            if epoch==0:
                                if 'jsa_labels' not in turn_batch:
                                    turn_batch['jsa_labels']=[]
                                turn_batch['jsa_labels'].append(jsa_labels)
                            else:
                                turn_batch['jsa_labels'][i]=jsa_labels
                            temp_label=copy.deepcopy(jsa_labels)
                            inputs_posterior=self.add_torch_input(temp_label[0])
                            labels_posterior=self.add_torch_input(temp_label[1])
                            inputs_prior=self.add_torch_input(temp_label[2])
                            labels_prior=self.add_torch_input(temp_label[3])
                            if epoch==0:
                                #straight through trick
                                ST_inputs_prior, resp_label=self.get_ST_input(inputs_prior['contexts_tensor'],\
                                    logits_post,labels_prior['contexts_tensor'],labels_posterior['contexts_tensor'])
                                embed_prior=ST_inputs_prior.matmul(self.PrioriModel.get_input_embeddings().weight)#multiple the input embedding
                                outputs1=self.PrioriModel(inputs_embeds=embed_prior)    
                                #outputs1=self.PrioriModel(inputs_prior['contexts_tensor'])
                                loss_pri=self.calculate_loss_and_accuracy(outputs1,labels_prior['contexts_tensor'])
                            else:
                                outputs1=self.PrioriModel(inputs_prior['contexts_tensor'].to(self.device1))
                                loss_pri=self.calculate_loss_and_accuracy(outputs1,labels_prior['contexts_tensor'].to(self.device1))
                                outputs2=self.PosteriorModel(inputs_posterior['contexts_tensor'].to(self.device2))
                                loss_pos=self.calculate_loss_and_accuracy(outputs2,labels_posterior['contexts_tensor'].to(self.device2))
                            
                            if cfg.loss_reg:
                                loss_pri=loss_pri/cfg.gradient_accumulation_steps
                                loss_pos=loss_pos/cfg.gradient_accumulation_steps
                            st3=0
                            loss_pri.backward()
                            if epoch!=0:
                                loss_pos.backward()
                                loss=loss_pri+loss_pos.to(self.device1)
                                tr_loss += loss_pri.item()+loss_pos.item()
                                uns_loss += loss_pri.item()+loss_pos.item()
                            else :
                                loss=loss_pri
                                tr_loss += loss_pri.item()
                                uns_loss += loss_pri.item()
                            uns_step+=1
                            torch.nn.utils.clip_grad_norm_(self.PrioriModel.parameters(), 5.0)
                            torch.nn.utils.clip_grad_norm_(self.PosteriorModel.parameters(), 5.0)
                        else:# supervised training
                            inputs_prior, labels_prior = self.reader.convert_batch_turn(batch, mini_pv_batch, first_turn, posterior=False)
                            inputs_posterior, labels_posterior = self.reader.convert_batch_turn(batch, mini_pv_batch, first_turn, posterior=True)
                            inputs_prior = self.add_torch_input(inputs_prior)#B,T
                            labels_prior=self.add_torch_input(labels_prior)#B,T
                            inputs_posterior=self.add_torch_input(inputs_posterior,posterior=True)
                            labels_posterior=self.add_torch_input(labels_posterior,posterior=True)

                            outputs1 = self.PrioriModel(inputs_prior['contexts_tensor'])
                            loss_pri=self.calculate_loss_and_accuracy(outputs1,labels_prior['contexts_tensor'])
                            outputs2=self.PosteriorModel(inputs_posterior['contexts_tensor'])
                            loss_pos=self.calculate_loss_and_accuracy(outputs2,labels_posterior['contexts_tensor'])

                            if cfg.loss_reg:
                                loss_pri=loss_pri/cfg.gradient_accumulation_steps
                                loss_pos=loss_pos/cfg.gradient_accumulation_steps
                            loss=loss_pri+loss_pos.to(self.device1)
                            loss.backward()
                            tr_loss+=loss.item()
                            sup_loss+=loss.item()
                            sup_step+=1
                            torch.nn.utils.clip_grad_norm_(self.PrioriModel.parameters(), 5.0)
                            torch.nn.utils.clip_grad_norm_(self.PosteriorModel.parameters(), 5.0)
                    epoch_step+=1
                    optimizer1.step()
                    optimizer1.zero_grad()
                    optimizer2.step()
                    optimizer2.zero_grad()
                    global_step+=1
                    if cfg.use_scheduler:
                        scheduler1.step()
                        scheduler2.step()
                    if self.tb_writer:
                        self.tb_writer.add_scalar('lr1', optimizer1.param_groups[0]["lr"],global_step)
                        self.tb_writer.add_scalar('lr2', optimizer2.param_groups[0]["lr"],global_step)
                        self.tb_writer.add_scalar('loss', loss.item(), global_step)
                    pv_batch=next_pv_batch
            
            logging.info('Epoch: {}, Train epoch time: {:.2f} min, loss:{:.3f}, avg_sup_loss:{:.3f}, avg_uns_loss:{:.3f}'.format(epoch, 
                (time.time()-btm)/60, tr_loss/epoch_step, sup_loss/(sup_step+1e-10), uns_loss/(uns_step+1e-10)))
            eval_result=self.validate_fast(data='dev')
            if self.tb_writer:
                self.tb_writer.add_scalar('joint_goal',eval_result['joint_acc'],epoch)
                self.tb_writer.add_scalar('match',eval_result['match'],epoch)
                self.tb_writer.add_scalar('success',eval_result['success'],epoch)
                self.tb_writer.add_scalar('bleu',eval_result['bleu'],epoch)
                self.tb_writer.add_scalar('combined_score',eval_result['score'],epoch)
            
            if eval_result['score']>max_score:
                max_score=eval_result['score']
                self.save_model(path='best_score_model')
                early_stop_count=cfg.early_stop_count
            else:
                weight_decay_count-=1
                if weight_decay_count==0 and not cfg.use_scheduler:
                    lr=lr*cfg.lr_decay
                    for group in optimizer1.param_groups:
                        group['lr'] = lr
                    for group in optimizer2.param_groups:
                        group['lr'] = lr
                    logging.info("learning rate decay to {}".format(lr))
                    weight_decay_count = cfg.weight_decay_count
                if epoch>=warmup_epochs:
                    early_stop_count-=1
                    logging.info('early stop count:%d'%early_stop_count)
            if lr<1e-9 and not cfg.use_scheduler:
                logging.info('learning rate too small, break')
                break
            if early_stop_count==0 and cfg.early_stop:
                logging.info('early stopped')
                break

    def get_kl_loss(self,logits_pri,logits_post,labels_pri,labels_post):
        # logits_pri:B,T1,V
        # logits_post:B,T2,V
        # labels_pri:B,T1. bspn's label in prior sequence
        # labels_post:B,T2. bspn's label in posterior sequence
        # what labels do is to find the logits corresponding to bspn
        loss=0
        count=0
        for dial_idx in range(logits_pri.size(0)):
            label_pri=labels_pri[dial_idx,:].ne(cfg.pad_id).long().cpu().tolist() #pad_id处为0，bspn为1
            label_post=labels_post[dial_idx,:].ne(cfg.pad_id).long().cpu().tolist()
            h_len=len(label_post)-label_post[::-1].index(1)-label_post.index(1)
            idx1=label_pri.index(1)
            idx2=label_post.index(1)
            probs_pri=F.softmax(logits_pri[dial_idx, idx1:idx1+h_len-1,:],dim=-1)
            probs_post=F.softmax(logits_post[dial_idx, idx2:idx2+h_len-1,:],dim=-1)
            loss+=self.kl_loss(probs_pri,probs_post.to(probs_pri.device))
            count+=h_len
        return loss/count


    def get_sep_optimizers(self, num_dials, model, num_batches=None):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": cfg.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            }
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=cfg.lr)
        if not num_batches:
            num_training_steps = num_dials*cfg.epoch_num // (cfg.gradient_accumulation_steps*cfg.origin_batch_size)
        else:
            num_training_steps = num_batches*cfg.epoch_num
        num_warmup_steps = cfg.warmup_steps if cfg.warmup_steps >= 0 else int(num_training_steps*cfg.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,\
            num_training_steps=num_training_steps) if cfg.use_scheduler else None
        logging.info('Training steps:{}, warmup steps:{}, steps per epoch:{}'.format(num_training_steps, 
            num_warmup_steps, num_batches))
        return optimizer, scheduler

    def add_torch_input(self, inputs, posterior=False):
        # to tensor and to device
        if 'contexts_np' not in inputs:
            inputs['contexts_np'],_=utils.padSeqs_gpt(inputs['contexts'], cfg.pad_id)
        contexts_tensor = torch.from_numpy(inputs['contexts_np']).long()
        if posterior:
            contexts_tensor = contexts_tensor.to(self.device2)
        else:
            contexts_tensor = contexts_tensor.to(self.device1)
        inputs['contexts_tensor'] = contexts_tensor
        return inputs


    def calculate_loss_and_accuracy(self, outputs, labels):
        # GPT2-chicahat/train.py
        lm_logits = outputs[0]

        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        pad_id = cfg.pad_id
        loss_fct = nn.CrossEntropyLoss(ignore_index=pad_id, reduction='sum')
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # avg loss
        not_ignore = shift_labels.ne(pad_id)
        num_targets = not_ignore.long().sum().item()

        loss /= num_targets
        return loss

    def get_max_len(self,batch):
        max_len=0
        for dial in batch:
            dial_len=0
            for turn in dial:
                dial_len+=len(turn['user'])+len(turn['resp'])
            if dial_len>max_len:
                max_len=dial_len
        return max_len
    

    def convert_eval_batch(self, data, contexts, turn_num,bs_gen,prior=False,db_gen=None,resp_gen=None,aspn_gen=None, gen_db=False):
        
        if gen_db:#在使用后验网络生成数据库结果时使用
            new_contexts=[]
            for id, context in enumerate(contexts):
                new_contexts.append(context[:-1] + bs_gen[id] + [self.sos_db_id])
            return new_contexts
        else:
            for id,context in enumerate(contexts):
                if turn_num==0:
                    if prior:
                        if db_gen is None:#还没有生成bs_gen以及对应的db_gen
                            contexts[id]=data[id][turn_num]['user']+[self.sos_b_id]
                        else:
                            sos_id=self.sos_a_id if cfg.model_act else self.sos_r_id
                            contexts[id]=context[:-1]+bs_gen[id]+db_gen[id]+[sos_id]
                    else:
                        if db_gen is None:
                            contexts[id]=data[id][turn_num]['user']+data[id][turn_num]['resp']+[self.sos_b_id]
                        else:
                            contexts[id]= context[:-1] + bs_gen[id]+ db_gen[id] + [self.sos_a_id]
                else:
                    #context中已经含有sos_b了
                    if prior:
                        if resp_gen is None:
                            sos_id=self.sos_a_id if cfg.model_act else self.sos_r_id
                            contexts[id]=context[:-1] +bs_gen[id]+db_gen[id]+[sos_id]
                        else:
                            contexts[id]=context[:-1] + resp_gen[id] + data[id][turn_num]['user']+[self.sos_b_id]
                    else:
                        if resp_gen is None:
                            if cfg.model_act:
                                contexts[id]=context[:-1] + bs_gen[id] + db_gen[id] + [self.sos_a_id]#to generate aspn
                            else:
                                contexts[id]=context[:-1] + bs_gen[id] +[self.sos_r_id]
                        else:
                            if cfg.model_act:
                                contexts[id]=context[:-1] + aspn_gen[id] + data[id][turn_num]['user']\
                                    +data[id][turn_num]['resp']+[self.sos_b_id]#to generate bspn
                            else:
                                contexts[id]=context[:-1]+data[id][turn_num]['user']+data[id][turn_num]['resp']+[self.sos_b_id]
            return contexts


    def get_bspn(self,bs_tensor, return_db=False, turn_domain=None):
        # return_db: return db results of bspn
        # turn_domain: a list of domain
        if not isinstance(bs_tensor,list):
            bs_batch=bs_tensor.cpu().tolist()
        else:
            bs_batch=bs_tensor
        bs_gen=[]
        db_gen=[]
        eos_b_id=self.eos_b_id
        sos_b_id=self.sos_b_id
        for i,bs in enumerate(bs_batch):
            if eos_b_id in bs:
                bs=[sos_b_id]+bs[:bs.index(eos_b_id)+1]
            else:
                bs[-1]=eos_b_id
                bs=[sos_b_id]+bs
            if bs.count(sos_b_id)>1:
                last=bs[::-1].index(sos_b_id)+1
                bs=bs[-last:]

            bs_gen.append(bs)
            if return_db:
                db_result=self.reader.bspan_to_DBpointer(self.tokenizer.decode(bs), turn_domain[i])
                db = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('<sos_db> '+ db_result + ' <eos_db>'))
                db_gen.append(db)
        if return_db:
            return bs_gen,db_gen
        else:
            return bs_gen

    def get_aspn(self,aspn_tensor):
        if not isinstance(aspn_tensor, list):
            aspn_batch=aspn_tensor.cpu().tolist()
        else:
            aspn_batch=aspn_tensor
        aspn_gen=[]
        eos_a_id=self.eos_a_id
        sos_a_id=self.sos_a_id
        for i ,aspn in enumerate(aspn_batch):
            if eos_a_id in aspn:
                aspn=[sos_a_id]+aspn[:aspn.index(eos_a_id)+1]
            else:
                aspn[-1]=eos_a_id
                aspn=[sos_a_id]+aspn
            if aspn.count(sos_a_id)>1:
                last=aspn[::-1].index(sos_a_id)+1
                aspn=aspn[-last:]
            aspn_gen.append(aspn)
        return aspn_gen

    def get_resp(self,resp_tensor):
        resp_batch=resp_tensor.cpu().tolist()
        resp_gen=[]
        eos_r_id=self.eos_r_id
        sos_r_id=self.sos_a_id if cfg.model_act else self.sos_r_id
        for i,resp in enumerate(resp_batch):
            if eos_r_id in resp:
                resp=[sos_r_id]+resp[:resp.index(eos_r_id)+1]
            else:
                resp[-1]=eos_r_id
                resp=[sos_r_id]+resp
            if resp.count(sos_r_id)>1:
                last=resp[::-1].index(sos_r_id)+1
                resp=resp[-last:]
            resp_gen.append(resp)
        return resp_gen
    
    def get_jsa_prob(self,logits_pri,logits_post,labels_pri,labels_post):
        # logits_pri:B,T1,V
        # logits_post:B,T2,V
        # labels_pri:B,T1. uspn's,bspn's label in prior sequence
        # labels_post:B,T2. bspn's label in posterior sequence
        # what labels do is to find the logits corresponding to bspn
        prob=[]
        for dial_idx in range(logits_pri.size(0)):
            label_pri=labels_pri[dial_idx,:].ne(cfg.pad_id).long().cpu().tolist() #pad_id处为0，bspn为1
            label_post=labels_post[dial_idx,:].ne(cfg.pad_id).long().cpu().tolist()
            h_len_post=len(label_post)-label_post[::-1].index(1)-label_post.index(1)
            h_len_pri=len(label_pri)-label_pri[::-1].index(1)-label_pri.index(1)
            idx1=label_pri.index(1)
            idx2=label_post.index(1)
            probs_pri=F.softmax(logits_pri[dial_idx, idx1:idx1+h_len_pri-1,:],dim=-1)
            probs_post=F.softmax(logits_post[dial_idx, idx2:idx2+h_len_post-1,:],dim=-1)
            up=torch.tensor(0.0)
            down=torch.tensor(0.0)
        
            for up_num in range(probs_pri.size()[0]):#loc2-loc1-1
                #if probs_pri.size()[0]!=loc2-loc1-1
                #    print(probs_pri.size()[0])
                #    print(loc2-loc1-1)
                up=up+math.log(probs_pri[up_num,labels_pri[dial_idx,idx1+up_num+1]])#probs_pri[up_num,:].max()
            for down_num in range(probs_post.size()[0]):#loc4-loc3-1
                down=down+math.log(probs_post[down_num,labels_post[dial_idx,idx2+down_num+1]])#probs_pri[down_num,labels_pri[logits_pri.size(1)-loc2+up_num]]
            prob.append(up.item()-down.item())
        return prob

    def get_turn_domain(self, turn_domain_batch, bs_batch, pv_bs_batch=None):

        db_batch=[]
        for i, bspn in enumerate(bs_batch):
            bspn_tokens=self.tokenizer.decode(bspn)
            cons=self.reader.bspan_to_constraint_dict(bspn_tokens)
            cur_domain=list(cons.keys())
            if len(cur_domain)==0:
                db_result = self.tokenizer.encode('<sos_db> [db_nores] <eos_db>')
            else:
                if len(cur_domain)==1:
                    turn_domain_batch[i]=cur_domain
                else:
                    if pv_bs_batch is None:
                        max_slot_num=0 # We choose the domain with most slots as the current domain
                        for domain in cur_domain:
                            if len(cons[domain])>max_slot_num:
                                turn_domain_batch[i]=[domain]
                                max_slot_num=len(cons[domain])
                    else:
                        pv_domain=list(self.reader.bspan_to_constraint_dict(self.tokenizer.decode(pv_bs_batch[i])).keys())
                        for domain in cur_domain:
                            if domain not in pv_domain: # new domain
                                # if domains are all the same, self.domain will not change
                                turn_domain_batch[i]=[domain]
                db_result = self.reader.bspan_to_DBpointer(bspn_tokens, turn_domain_batch[i]) #[db_x]
                db_result = self.tokenizer.encode('<sos_db> '+ db_result + ' <eos_db>')
            db_batch.append(db_result)
        return turn_domain_batch, db_batch

    def save_model(self, posterior=False, path=None, model=None):
        if not path:
            if posterior:
                save_path = os.path.join(cfg.exp_path, 'best_model_post')
            else:
                save_path = os.path.join(cfg.exp_path, 'best_model_pri')
        else:
            save_path = os.path.join(cfg.exp_path, path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        logging.info('Saving model checkpoint to %s', save_path)
        if not model:
            if posterior:
                self.PosteriorModel.save_pretrained(save_path)
            else:
                self.PrioriModel.save_pretrained(save_path)
        else:
            model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        # save cfg
    
    def eval(self,data='dev',posterior=False,model=None):
        model.eval()
        all_batches = self.reader.get_batches(data)
        total_batch=len(all_batches)
        total_loss=0
        with torch.no_grad():
            data_iterator = self.reader.get_data_iterator(all_batches)
            for batch_idx, dial_batch in enumerate(data_iterator):
                pv_batch = None
                for turn_num, turn_batch in enumerate(dial_batch):
                    first_turn = (turn_num == 0)
                    inputs, labels = self.reader.convert_batch_turn(turn_batch, pv_batch, first_turn, posterior=posterior)
                    pv_batch = self.reader.get_pv_batch(pv_batch, turn_batch['user'],
                        turn_batch['resp'], turn_batch['bspn'])
                    inputs=self.add_torch_input(inputs)#B,T
                    labels=self.add_torch_input(labels)#B,T
                    outputs = model(inputs['contexts_tensor'])
                    loss=self.calculate_loss_and_accuracy(outputs,labels['contexts_tensor'])
                    total_loss+=loss.item()
        return total_loss/total_batch


    def validate_fast(self, data='dev', dial_id_list=None):
        if cfg.mode=='pretrain' or cfg.mode=='train' or cfg.mode=='semi_ST':
            self.PrioriModel=self.model
            self.device1=self.model.device
        
        self.PrioriModel.eval()
        eval_data = self.reader.get_eval_data(data)
        #if cfg.debugging:
         #   eval_data=eval_data[:32]
        origin_batch_size=cfg.batch_size
        cfg.batch_size=cfg.eval_batch_size
        batches=self.reader.get_batches('test',data=eval_data)
        result_path=os.path.join(cfg.eval_load_path,'result.json')
        
        if os.path.exists(result_path) and cfg.mode=='test':
            #results,field=self.reader.load_result(result_path)
            results=json.load(open(result_path, 'r'))
            joint_acc=compute_jacc(results, mode='JointGoalAcc')
            joint_acc=joint_acc['precision']
            eval_st=time.time()
            bleu, success, match = self.evaluator.validation_metric(results)
            logging.info('Evaluation time:{:.2f}s'.format(time.time()-eval_st))
            score = 0.5 * (success + match) + bleu
            logging.info('Validation %2.2f  %2.2f  %2.2f  %.2f  %.3f' % (match, success, bleu, score, joint_acc))
            eval_results = {}
            eval_results['bleu'] = bleu
            eval_results['success'] = success
            eval_results['match'] = match
            eval_results['score'] = score
            eval_results['joint_acc']=joint_acc
            eval_results['result'] = 'validation [CTR] match: %2.2f  success: %2.2f  bleu: %2.2f    score: %.2f' % (match, success, bleu, score)
            return eval_results
        
        # valid_losses = []
        result_collection = {}
        st=time.time()
        for batch in batches:
            try:
                if batch==[]:
                    continue
                if cfg.turn_level:
                    batch=self.generate_batch_turn_level(batch)
                else:
                    batch=self.generate_batch_session_level(batch)
                for dialog in batch:
                    result_collection.update(self.reader.inverse_transpose_turn(dialog))
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    logging.info("WARNING: ran out of memory during validation and batch will be divided by half, batch size:{}, turn num:{}"\
                        .format(len(batch),len(batch[0])))
                    if hasattr(torch.cuda, 'empty_cache'):
                        with torch.cuda.device(self.device1):
                            torch.cuda.empty_cache()
                    #divide the batch in half if out of memory
                    batches.insert(0,batch[:len(batch)//2])
                    batches.insert(1,batch[len(batch)//2:])
                else:
                    logging.info(str(exception))
                    raise exception
        results, field = self.reader.wrap_result_lm(result_collection)
        logging.info('Inference time:{:.3f} min'.format((time.time()-st)/60))

        joint_acc=compute_jacc(results, mode='JointGoalAcc')
        joint_acc=joint_acc['precision']
        bleu, success, match = self.evaluator.validation_metric(results)
        score = 0.5 * (success + match) + bleu
        logging.info('Validation %2.2f  %2.2f  %2.2f  %.2f  %.3f' % (match, success, bleu, score, joint_acc))
        json.dump(results, open(result_path, 'w'), indent=2, ensure_ascii=False)
        #self.reader.save_result('w', results, field,result_name='result.csv')

        eval_results = {}
        eval_results['bleu'] = bleu
        eval_results['success'] = success
        eval_results['match'] = match
        eval_results['score'] = score
        eval_results['joint_acc']=joint_acc
        cfg.batch_size=origin_batch_size
        return eval_results


    def generate_batch(self, model, contexts, max_len, eos_id, beam=1):
        # generate by batch
        # contexts: a list of ids
        # max_len: the max generated length
        # eos_id: the end id
        # return: a batch of ids with pre pad 
        batch_size=len(contexts)
        end_flag=np.zeros(batch_size)
        if beam>1:
            beam_box=[beam]*batch_size
            beam_result=[[] for _ in range(batch_size)]
            max_prob=[-float('inf')]*batch_size
        past_key_values=None
        inputs,attentions=self.reader.batch_align(contexts,left_len=max_len,return_attn=True)
        inputs=torch.tensor(inputs).to(model.device)
        attentions=torch.tensor(attentions).to(model.device)
        model.eval()
        with torch.no_grad():
            for i in range(max_len):
                if beam==1:
                    position_ids = attentions.long().cumsum(-1) - 1
                    position_ids.masked_fill_(attentions == 0, 1)
                    if past_key_values is not None:
                        position_ids=position_ids[:, -1].unsqueeze(-1)
                    if inputs.size(0)==0:
                        raise ValueError(contexts, inputs.cpu().list(), attentions)
                    outputs=model(inputs,attention_mask=attentions,position_ids=position_ids,\
                            return_dict=True,use_cache=True,past_key_values=past_key_values)

                    past_key_values=outputs.past_key_values

                    preds=outputs.logits[:,-1,:].argmax(-1)#B
                    if i==0:
                        gen_tensor=preds.unsqueeze(1)
                    else:
                        gen_tensor=torch.cat([gen_tensor,preds.unsqueeze(1)],dim=1)
                    attentions=torch.cat((attentions,torch.ones(batch_size,1).long().to(model.device)),dim=1)
                    inputs=preds.unsqueeze(1)
                    end_flag+=(preds.cpu().numpy()==eos_id).astype(float)
                    if sum(end_flag==0)==0:
                        break
                else:
                    if i==0:
                        position_ids = attentions.long().cumsum(-1) - 1
                        position_ids.masked_fill_(attentions == 0, 1)
                        outputs=model(inputs,attention_mask=attentions,position_ids=position_ids,\
                                return_dict=True,use_cache=True,past_key_values=past_key_values)
                        past_key_values=[outputs.past_key_values]*beam
                        log_prob=F.log_softmax(outputs.logits[:, -1, :], -1) # B, V
                        beam_prob, beam_idx=torch.topk(log_prob, beam, -1) # B, beam
                        gen_tensor=beam_idx.unsqueeze(-1)# B, beam, 1
                        attentions=torch.cat((attentions,torch.ones(batch_size,1).long().to(model.device)),dim=1)
                        position_ids = attentions.long().cumsum(-1) - 1
                        position_ids.masked_fill_(attentions == 0, 1)
                        position_ids=position_ids[:, -1].unsqueeze(-1)
                        pv_beam_prob=beam_prob #B, beam
                        pv_beam_idx=beam_idx#B, beam
                    else:
                        for j in range(beam):
                            inputs=pv_beam_idx[:,j].unsqueeze(-1) # B, 1
                            outputs=model(inputs,attention_mask=attentions,position_ids=position_ids,\
                                return_dict=True,use_cache=True,past_key_values=past_key_values[j])
                            past_key_values[j]=outputs.past_key_values
                            log_prob=F.log_softmax(outputs.logits[:, -1, :], -1) # B, V
                            beam_prob, beam_idx=torch.topk(log_prob, beam, -1) # B, beam
                            if j==0:
                                prob_pool= beam_prob+pv_beam_prob[:, j].unsqueeze(-1).expand(-1, beam) # B, beam
                                id_pool=beam_idx
                            else:
                                prob_pool=torch.cat([prob_pool, beam_prob+pv_beam_prob[:, j].unsqueeze(-1).expand(-1, beam)],-1) # B, beam*beam
                                id_pool=torch.cat([id_pool, beam_idx], -1)# B, beam*beam
                        beam_prob, temp_id=torch.topk(prob_pool, beam, -1) #B, beam
                        beam_idx=torch.gather(id_pool, -1, temp_id)
                        temp_id=temp_id//beam
                        new_past_key_values=copy.deepcopy(past_key_values)
                        for b in range(batch_size):
                            gen_tensor[b, :, :]=gen_tensor[b, :, :].index_select(0, temp_id[b, :])
                            for t in range(beam):
                                for l in range(6):
                                    new_past_key_values[t][l][:, b, :,:,:]=past_key_values[temp_id[b, t]][l][:, b, :, :, :]
                        past_key_values=new_past_key_values
                        #past_key_values=[past_key_values[t] for t in temp_id.cpu().list()]
                        gen_tensor=torch.cat([gen_tensor, beam_idx.unsqueeze(-1)],-1) #B, beam, T
                        attentions=torch.cat((attentions,torch.ones(batch_size,1).long().to(model.device)),dim=1)
                        position_ids = attentions.long().cumsum(-1) - 1
                        position_ids.masked_fill_(attentions == 0, 1)
                        position_ids=position_ids[:, -1].unsqueeze(-1)
                        pv_beam_prob=beam_prob #B, beam
                        pv_beam_idx=beam_idx
                    for m in range(batch_size):
                        for n, gen in enumerate(gen_tensor.cpu().tolist()[m]):
                            if eos_id in gen:
                                beam_box[m]-=1
                                avg_prob=pv_beam_prob[m][n]/len(gen)
                                beam_result[m].append((gen, avg_prob))
                                pv_beam_prob[m][n]=-float('inf')
                    # we do not break during beam search
                    #if not any(beam_box):
                     #   break
            
        if beam==1:
            return gen_tensor.cpu().tolist()
        else:
            for i, tup in enumerate(beam_result):
                beam_list=sorted(tup, key=lambda item:item[1], reverse=True)
                beam_result[i]=[item[0] for item in beam_list[:beam]]
            return beam_result        

    
    def generate_batch_turn_level(self, batch):
        
        batch=self.reader.transpose_batch(batch)

        bs_max_len=75
        resp_max_len=80
        sos_b_id=self.sos_b_id
        eos_b_id=self.eos_b_id
        sos_r_id=self.sos_r_id
        eos_a_id=self.eos_a_id
        eos_r_id=self.eos_r_id

        batch_size=len(batch[0]['dial_id'])
        contexts=[[] for i in range(batch_size)]
        bs_gen=[]
        db_gen=[]
        resp_gen=[]
        pv_batch=None
        pv_bspn_batch=None
        turn_domain_batch=[[] for _ in range(batch_size)]

        device=self.device1
        with torch.no_grad():
            for turn_num, turn_batch in enumerate(batch):
                # generate bspn
                past_key_values=None
                end_flag=np.zeros(batch_size)
                contexts=self.reader.convert_eval_batch_turn(turn_batch,pv_batch, mode='gen_bspn')
                
                if self.global_output>0 and cfg.mode=='test':
                    logging.info(self.tokenizer.decode(contexts[0]))
                    self.global_output-=1
                
                inputs,attentions=self.reader.batch_align(contexts,left_len=bs_max_len,return_attn=True)
                inputs=torch.tensor(inputs).to(device)
                attentions=torch.tensor(attentions).to(device)
                for i in range(bs_max_len):
                    position_ids = attentions.long().cumsum(-1) - 1
                    position_ids.masked_fill_(attentions == 0, 1)
                    if past_key_values is not None:
                        position_ids=position_ids[:, -1].unsqueeze(-1)
                    outputs=self.PrioriModel(inputs,attention_mask=attentions,position_ids=position_ids,\
                            return_dict=True,use_cache=True,past_key_values=past_key_values)

                    past_key_values=outputs.past_key_values

                    preds=outputs.logits[:,-1,:].argmax(-1)#B
                    if i==0:
                        bs_tensor=preds.unsqueeze(1)
                    else:
                        bs_tensor=torch.cat([bs_tensor,preds.unsqueeze(1)],dim=1)
                    attentions=torch.cat((attentions,torch.ones(batch_size,1).long().to(device)),dim=1)
                    inputs=preds.unsqueeze(1)
                    end_flag+=(preds.cpu().numpy()==eos_b_id).astype(float)
                    if sum(end_flag==0)==0:
                        break
                if cfg.use_true_domain_for_ctr_eval:
                    bs_gen,db_gen=self.get_bspn(bs_tensor,return_db=True, turn_domain=turn_batch['turn_domain'])
                else:
                    bs_gen=self.get_bspn(bs_tensor)
                    turn_domain_batch, db_gen=self.get_turn_domain(turn_domain_batch, bs_gen, pv_bspn_batch)
                # generate aspn and resp
                past_key_values=None
                end_flag=np.zeros(batch_size)
                contexts=self.reader.convert_eval_batch_turn(turn_batch,pv_batch, mode='gen_ar', 
                    bspn_gen=bs_gen,db_gen=db_gen)
                
                #if self.global_output>0 and cfg.mode=='test':
                 #   logging.info(self.tokenizer.decode(contexts[0]))
                inputs,attentions=self.reader.batch_align(contexts,left_len=resp_max_len,return_attn=True)
                inputs=torch.tensor(inputs).to(device)
                attentions=torch.tensor(attentions).to(device)
                for i in range(resp_max_len):
                    position_ids = attentions.long().cumsum(-1) - 1
                    position_ids.masked_fill_(attentions == 0, 1)
                    if past_key_values is not None:
                        position_ids=position_ids[:, -1].unsqueeze(-1)
                    outputs=self.PrioriModel(inputs,attention_mask=attentions,position_ids=position_ids,\
                            return_dict=True,use_cache=True,past_key_values=past_key_values)
                    past_key_values=outputs.past_key_values
                    preds=outputs.logits[:,-1,:].argmax(-1)#B
                    if i==0:
                        resp_tensor=preds.unsqueeze(1)
                    else:
                        resp_tensor=torch.cat([resp_tensor,preds.unsqueeze(1)],dim=1)
                    attentions=torch.cat((attentions,torch.ones(batch_size,1).long().to(device)),dim=1)
                    inputs=preds.unsqueeze(1)
                    end_flag+=(preds.cpu().numpy()==eos_r_id).astype(float)
                    if sum(end_flag==0)==0:
                        break
                resp_gen=self.get_resp(resp_tensor)
                aspn_gen=[]
                for i, temp in enumerate(resp_gen):
                    if eos_a_id in temp:
                        aspn=temp[:temp.index(eos_a_id)+1]
                    else:
                        aspn=temp[:-1]+[eos_a_id]
                    if sos_r_id in temp:
                        resp=temp[temp.index(sos_r_id):]
                    else:
                        resp=[sos_r_id]+temp[1:]
                    resp_gen[i]=resp
                    aspn_gen.append(aspn)
                pv_batch=self.reader.get_pv_batch(pv_batch, turn_batch['user'], resp_gen, bs_gen)
                turn_batch['bspn_gen']=bs_gen
                turn_batch['aspn_gen']=aspn_gen
                turn_batch['resp_gen']=resp_gen
                turn_batch['db_gen']=db_gen
                pv_bspn_batch=bs_gen
        return self.reader.inverse_transpose_batch(batch)
    

def parse_arg_cfg(args):
    # add args to cfg
    if args.cfg:
        for pair in args.cfg:
            k, v = tuple(pair.split('='))
            dtype = type(getattr(cfg, k))
            if dtype == type(None):
                raise ValueError()
            if dtype is bool:
                v = False if v == 'False' else True
            elif dtype is list:
                v = v.split(',')
                if k == 'cuda_device':
                    v = [int(no) for no in v]
            else:
                v = dtype(v)
            setattr(cfg, k, v)
    return


def main():
    if not os.path.exists('./experiments'):
        os.mkdir('./experiments')

    parser = argparse.ArgumentParser()
    parser.add_argument('-mode')
    parser.add_argument('-cfg', nargs='*')
    args = parser.parse_args()

    cfg.mode = args.mode
    if 'test' in args.mode:
        parse_arg_cfg(args)
        cfg.eval_load_path=cfg.gpt_path
    else:  # train
        parse_arg_cfg(args)
        #print('exp_no:',cfg.exp_no)
        if cfg.exp_path in ['', 'to be generated']:
            experiments_path = './experiments_21' if cfg.dataset==1 else './experiments'
            if cfg.exp_no=='':
                if cfg.mode=='pretrain':
                    if cfg.posterior_train:
                        cfg.exp_no='pre_pos'
                    else:
                        cfg.exp_no='pre_'
                elif cfg.mode=='semi_ST':
                    cfg.exp_no='ST_'
                    if cfg.fix_ST:
                        cfg.exp_no=cfg.exp_no+'fix_'
                elif cfg.mode=='semi_VL':
                    cfg.exp_no='VL_'
                elif cfg.mode=='train':
                    cfg.exp_no='full'
                    if cfg.posterior_train:
                        cfg.exp_no = cfg.exp_no + '_pos'
                if cfg.mode!='train':
                    cfg.exp_no = cfg.exp_no + str(cfg.spv_proportion)
                if cfg.model_act:
                    cfg.exp_no = cfg.exp_no + '_act'
                if cfg.data_aug:
                    cfg.exp_no='full_aug_VL' if cfg.mode=='semi_VL' else 'full_aug_ST'
            print('exp_no:',cfg.exp_no)
            cfg.exp_path = os.path.join(experiments_path,'{}_{}_sd{}_lr{}_bs{}_ga{}'.format('-'.join(cfg.exp_domains),
                                                                          cfg.exp_no, cfg.seed, cfg.lr, cfg.batch_size,
                                                                          cfg.gradient_accumulation_steps))
            if 'test' not in cfg.mode:
                print('save path:', cfg.exp_path)
            if cfg.save_log:
                if not os.path.exists(cfg.exp_path):
                    os.mkdir(cfg.exp_path)

            # to gpt later
            cfg.model_path = os.path.join(cfg.exp_path, 'model.pkl')
            cfg.result_path = os.path.join(cfg.exp_path, 'result.csv')
            cfg.vocab_path_eval = os.path.join(cfg.exp_path, 'vocab')
            cfg.eval_load_path = cfg.exp_path
    cfg._init_logging_handler(args.mode)
    device=cfg.cuda_device
    cfg.divided_path=os.path.join(cfg.data_path,'divided_data{}.json'.format(cfg.spv_proportion))

    # fix random seed
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    # initialize model
    m = Modal(device)

    if args.mode =='pretrain' or args.mode=='train':
        m.pretrain_turn_level(posterior=cfg.posterior_train)
    elif args.mode =='semi_VL':
        m.semi_VL()
    elif args.mode == 'semi_ST':
        m.semi_ST()
    elif args.mode=='semi_jsa':
        m.semi_jsa()
    else:  # test
        logging.info('Load model from :{}'.format(cfg.eval_load_path))
        m.validate_fast('test')

if __name__ == "__main__":
    main()
