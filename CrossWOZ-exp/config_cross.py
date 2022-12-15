import logging, time, os

class _Config:
    def __init__(self):
        self._crosswoz_damd_init()

    def _crosswoz_damd_init(self):

        self.vocab_path_train = './data/CrossWoz-processed/vocab'
        self.vocab_path_eval = None
        self.data_path = './data/CrossWoz-processed/'
        self.data_file = 'data_for_damd.json'
        self.dev_list = 'data/CrossWOZ/valListFile.json'
        self.test_list = 'data/CrossWOZ/testListFile.json'
        self.dbs = {
            'attraction': 'database/attraction_db.json',
            'hotel': 'database/hotel_db.json',
            'metro': 'database/metro_db.json',
            'restaurant': 'database/restaurant_db.json',
            'taxi': 'database/taxi_db.json',
        }
        
        self.glove_path = './data/glove/glove.6B.50d.txt'
        self.domain_file_path = 'data/CrossWoz-processed/domain_files.json'
        self.slot_value_set_path = 'database/value_set.json'#need to be generated
        self.multi_acts_path = 'data/CrossWoz-processed/multi_act_mapping_train.json'#need to be generated
        self.exp_path = 'to be generated'
        self.log_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

        self.turn_level=True # turn-level training or session-level training
        self.input_history=False # whether or not add the whole dialog history into the training sequence if train with turn-level 
        self.input_prev_resp=True # whether or not add the prev response into the training sequence if input_history is False
        self.only_target_loss=True

        #key training settings
        self.spv_proportion=50
        self.model_act=True
        self.save_type='max_score'#'min_loss'
        #self.save_type='min_loss'#'max_score'
        self.early_stop=False
        self.mixed_train=False
        self.dataset=0
        self.example_log=True
        self.delex_as_damd = True 
        self.gen_db=False #critical setting. Only True when we use posterior model to generate belief state.
        #pretrain:
        self.posterior_train=False
        #VLtrain:
        self.VL_with_kl=True 
        self.PrioriModel_path='to be generated'
        self.PosteriorModel_path='to be generated'
        #STtrain:
        self.fix_ST=True # whether add straight through trick
        self.ST_resp_only=True #whether calculate cross-entropy on response only
        #evaluation:
        self.fast_validate=True
        self.eval_batch_size=32
        self.gpt_path = "uer/gpt2-chinese-cluecorpussmall"
        self.val_set='test'
        self.col_samples=False #collect wrong predictions samples
        self.test_data_path=''
        #additional data setting
        self.data_aug=False
        self.only_SGD=False
        self.only_TM=False
        self.len_limit=True

        self.use_true_bspn_for_ctr_eval = False
        self.use_true_domain_for_ctr_eval = True
        self.use_true_domain_for_ctr_train = True

        self.post_loss_weight=0.5
        self.kl_loss_weight=0.5
        self.debugging=False
        

        self.loss_reg=False
        self.divided_path='to be generated'
        self.gradient_checkpoint=False
        self.fix_loss=False

        self.sample_type='top1'#'topk'
        self.topk_num=10#only when sample_type=topk
        

        # experiment settings
        self.mode = 'unknown'
        self.cuda = True
        self.cuda_device = [0]
        self.exp_no = ''
        self.seed = 11
        self.save_log = True # tensorboard 
        self.evaluate_during_training = True # evaluate during training
        self.truncated = False

        # training settings
        self.lr = 1e-4
        self.warmup_steps = -1 
        self.warmup_ratio= 0.2
        self.weight_decay = 0.0 
        self.gradient_accumulation_steps = 16
        self.batch_size = 2

        self.lr_decay = 0.5
        self.use_scheduler=True
        self.epoch_num = 40
        self.early_stop=False
        self.early_stop_count = 4
        self.weight_decay_count = 2

        # evaluation settings
        self.eval_load_path = 'to be generated'
        self.model_output = 'to be generated'
        self.eval_per_domain = False

        self.use_true_prev_bspn = False
        self.use_true_prev_aspn = False
        self.use_true_db_pointer = False
        self.use_true_prev_resp = False

        self.use_true_curr_bspn = False
        self.use_true_curr_aspn = False
        self.use_all_previous_context = True


        self.exp_domains = ['all'] # hotel,train, attraction, restaurant, taxi
        self.log_path = ''
        self.low_resource = False


        # model settings
        self.vocab_size = 3000
        self.enable_aspn = True
        self.enable_bspn = True
        self.bspn_mode = 'bspn' # 'bspn' or 'bsdx'
        self.enable_dspn = False # removed
        self.enable_dst = False
        #useless settings
        self.multi_acts_training = False
        self.same_eval_as_cambridge = True
        self.same_eval_act_f1_as_hdsa = False


    def __str__(self):
        s = ''
        for k,v in self.__dict__.items():
            s += '{} : {}\n'.format(k,v)
        return s


    def _init_logging_handler(self, mode):
        stderr_handler = logging.StreamHandler()
        if not os.path.exists('./log'):
            os.mkdir('./log')
        if self.save_log and ('train' in mode or 'semi' in mode):
            if self.dataset==0:
                file_handler = logging.FileHandler('./log/log_{}_{}_sd{}.txt'.format(mode, self.exp_no, self.seed))
            elif self.dataset==1:
                file_handler = logging.FileHandler('./log21/log_{}_{}_sd{}.txt'.format(mode, self.exp_no, self.seed))
        elif 'test' in mode and os.path.exists(self.eval_load_path):
            eval_log_path = os.path.join(self.eval_load_path, 'eval_log.json')
            # if os.path.exists(eval_log_path):
            #     os.remove(eval_log_path)
            file_handler = logging.FileHandler(eval_log_path)
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(handlers=[stderr_handler, file_handler])
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

global_config = _Config()

