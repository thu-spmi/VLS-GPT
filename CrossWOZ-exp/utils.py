import logging
import json,zipfile
import numpy as np
from collections import OrderedDict
import ontology_cross as ontology
from tqdm import tqdm
oog_list=['\u200e','\xc7','\u2615','\ufe0f' ,'\xe7' ,'\u200d' ,'\u3e06' ,'\u2022','\u3711' ,'\u2795','\xa0' ]
error_word='*错误字符*'
def generate_value_set():
    value_set={}
    
    for domain in ontology.db_domains_cross: # ['restaurant', 'hotel', 'attraction', 'taxi', 'metro']
        value_dict={}
        data = json.loads(open('database/'+domain+'_db.json', 'r', encoding='utf-8').read().lower())
        for obj in data:
            for key,value in obj[1].items():
                if key not in value_dict:
                    value_dict[key]=[value]
                else:
                    if value not in value_dict[key]:
                        value_dict[key].append(value)
        value_set[ontology.domains_cross_switch[domain]]=value_dict
        for key,values in value_dict.items():
            deleteList=[]
            for value in values:
                if value and isinstance(value,str):
                    for oog_word in oog_list:
                        if oog_word in value:
                            value.replace(oog_word,error_word)
                if value and isinstance(value,list):
                    subdeleteList=[]
                    for subvalue in value:
                        if subvalue and isinstance(subvalue,str):
                            for oog_word in oog_list:
                                if oog_word in subvalue:
                                    subvalue.replace(oog_word,error_word)
                    for subdeleting in subdeleteList:
                        value.remove(subdeleting)
            for deleting in deleteList:
                values.remove(deleting)
    with open('database/value_set.json', 'w') as f:
        json.dump(value_set, f, indent=2, ensure_ascii=False)    

def get_data_related():
    data_path = 'data/CrossWOZ/'
    #archive = zipfile.ZipFile(data_path + 'train.json.zip', 'r')
    data_all = json.loads(open(data_path + 'train.json', 'r').read().lower())
    #archive = zipfile.ZipFile(data_path + 'test.json.zip', 'r')
    test_data = json.loads(open(data_path + 'test.json', 'r').read().lower())
    #archive = zipfile.ZipFile(data_path + 'val.json.zip', 'r')
    val_data = json.loads(open(data_path + 'val.json', 'r').read().lower())
    data_all.update(test_data)
    data_all.update(val_data)
    with open(data_path +'testListFile.json', 'w') as f:
        for temp in test_data.keys():
            f.writelines(temp+'\n')
    with open(data_path +'valListFile.json', 'w') as f:
        for temp in val_data.keys():
            f.writelines(temp+'\n') 
    
    for fn, raw_dial in tqdm(list(data_all.items())): 
        for key,values in raw_dial.items():
            if values and isinstance(values,list):
                for temp in range(len(values)):
                    if values[temp] and isinstance(values[temp],str):
                        for oog_word in oog_list:
                            if oog_word in values[temp]:
                                values[temp]=values[temp].replace(oog_word,error_word)
                    if values[temp] and isinstance(values[temp],list):
                        deleteList1=[]
                        for temp1 in range(len(values[temp])):
                            if values[temp][temp1] and isinstance(values[temp][temp1],str):
                                for oog_word in oog_list:
                                    if oog_word in values[temp][temp1]:
                                        values[temp][temp1]=values[temp][temp1].replace(oog_word,error_word)
                            if values[temp][temp1] and isinstance(values[temp][temp1],list):
                                for temp2 in range(len(values[temp][temp1])):
                                    if values[temp][temp1][temp2] and isinstance(values[temp][temp1][temp2],str):
                                        for oog_word in oog_list:
                                            if oog_word in values[temp][temp1][temp2]:
                                                values[temp][temp1][temp2]=values[temp][temp1][temp2].replace(oog_word,error_word)
                    if values[temp] and isinstance(values[temp],dict):
                        deleteList=[]
                        for key,subvalue1 in values[temp].items():
                            if subvalue1 and isinstance(subvalue1,str):
                               for oog_word in oog_list:
                                    if oog_word in subvalue1:
                                        deleteList.append((key,oog_word))
                            if subvalue1 and isinstance(subvalue1,list):
                                for temp3 in range(len(subvalue1)):
                                    if subvalue1[temp3] and isinstance(subvalue1[temp3],list):
                                        for temp4 in range(len(subvalue1[temp3])):
                                            if subvalue1[temp3][temp4] and isinstance(subvalue1[temp3][temp4],str):
                                                for oog_word in oog_list:
                                                    if oog_word in subvalue1[temp3][temp4]:
                                                        values[temp][key][temp3][temp4]=values[temp][key][temp3][temp4].replace(oog_word,error_word)
                                            if subvalue1[temp3][temp4] and isinstance(subvalue1[temp3][temp4],list):
                                                for temp5 in range(len(subvalue1[temp3][temp4])):
                                                    if subvalue1[temp3][temp4][temp5] and isinstance(subvalue1[temp3][temp4][temp5],str):
                                                        for oog_word in oog_list:
                                                            if oog_word in subvalue1[temp3][temp4][temp5]:
                                                                values[temp][key][temp3][temp4][temp5]=values[temp][key][temp3][temp4][temp5].replace(oog_word,error_word)
                            if subvalue1 and isinstance(subvalue1,dict):
                                for key1,subvalue2 in subvalue1.items():
                                    if subvalue2 and isinstance(subvalue2,dict):
                                        deleteList1=[]
                                        for key2,subvalue3 in subvalue2.items():
                                            if subvalue3 and isinstance(subvalue3,str):
                                                for oog_word in oog_list:
                                                    if oog_word in subvalue3:
                                                        deleteList1.append((key2,oog_word))
                                            if subvalue3 and isinstance(subvalue3,list):
                                                for temp6 in range(len(subvalue3)):
                                                    if subvalue3[temp6] and isinstance(subvalue3[temp6],str):
                                                        for oog_word in oog_list:
                                                            if oog_word in subvalue3[temp6]:
                                                                values[temp][key][key1][key2][temp6]=values[temp][key][key1][key2][temp6].replace(oog_word,error_word)
                                        for temp7 in deleteList1:
                                            values[temp][key][key1][temp7[0]]=values[temp][key][key1][temp7[0]].replace(temp7[1],error_word)
                        for temp8 in deleteList:
                            values[temp][temp8[0]]=values[temp][temp8[0]].replace(temp8[1],error_word)
    with open(data_path +'data.json', 'w') as f:
        #json.dump(data_all, f, indent=2,ensure_ascii=False) 
        json.dump(data_all, f, indent=2, ensure_ascii=False)   

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        # torch.topk()返回最后一维最大的top_k个元素，返回值为二维(values,indices)
        # ...表示其他维度由计算机自行推断
        indices_to_remove = logits < torch.topk(logits, top_k)[
            0][..., -1, None]
        logits[indices_to_remove] = filter_value  # 对于topk之外的其他元素的logits值设为负无穷

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(
            logits, descending=True)  # 对logits进行递减排序
        cumulative_probs = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[...,
                                 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

def py2np(list):
    return np.array(list)


def write_dict(fn, dic):
    with open(fn, 'w') as f:
        json.dump(dic, f, indent=2)

def write_dict_cross(fn, dic):
    with open(fn, 'w') as f:
        count = 0
        deleteList=[]
        for key,value in dic.items():
            if ('\u200e' in key) or ('\xc7' in key)or ('\u2615' in key)or ('\ufe0f' in key)or ('\u200d' in key)or ('\xe7' in key)or ('\u3e06' in key)or ('\u2022' in key)or ('\u3711' in key)or ('\u2795' in key):
                deleteList.append(key)
        for deleting in deleteList:
            dic.pop(deleting)
        
        #json.dump(dic, f, indent=2,ensure_ascii=False)
        json.dump(dic, f, indent=2)


def f1_score(label_list, pred_list):
    tp = len([t for t in pred_list if t in label_list])
    fp = max(0, len(pred_list) - tp)
    fn = max(0, len(label_list) - tp)
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    return f1

class Vocab(object):
    def __init__(self, vocab_size=0):#can be modified to suit chinese environment
        self.vocab_size = vocab_size
        self.vocab_size_oov = 0   # get after construction
        self._idx2word = {}   #word + oov
        self._word2idx = {}   # word
        self._freq_dict = {}   #word + oov
        for w in ['<pad>', '<go_r>', '<unk>', '<go_b>', '<go_a>','<eos_u>', '<eos_r>',
                      '<eos_b>', '<eos_a>', '<go_d>','<eos_d>']:
            self._absolute_add_word(w)

    def _absolute_add_word(self, w):
        idx = len(self._idx2word)
        self._idx2word[idx] = w
        self._word2idx[w] = idx

    def add_word(self, word):
        if word not in self._freq_dict:
            self._freq_dict[word] = 0
        self._freq_dict[word] += 1

    def has_word(self, word):
        return self._freq_dict.get(word)

    def _add_to_vocab(self, word):
        if word not in self._word2idx:
            idx = len(self._idx2word)
            self._idx2word[idx] = word
            self._word2idx[word] = idx

    def construct(self):
        l = sorted(self._freq_dict.keys(), key=lambda x: -self._freq_dict[x])
        print('Vocabulary size including oov: %d' % (len(l) + len(self._idx2word)))
        if len(l) + len(self._idx2word) < self.vocab_size:
            logging.warning('actual label set smaller than that configured: {}/{}'
                            .format(len(l) + len(self._idx2word), self.vocab_size))
        for word in ontology.all_domains + ['general']:
            word = '[' + word + ']'
            self._add_to_vocab(word)
        for word in ontology.all_acts:
            word = '[' + word + ']'
            self._add_to_vocab(word)
        for word in ontology.all_slots:
            self._add_to_vocab(word)
        for word in l:
            if word.startswith('[value_') and word.endswith(']'):
                self._add_to_vocab(word)
        for word in l:
            self._add_to_vocab(word)
        self.vocab_size_oov = len(self._idx2word)
    
    def construct_cross(self):
        l = sorted(self._freq_dict.keys(), key=lambda x: -self._freq_dict[x])
        print('Vocabulary size including oov: %d' % (len(l) + len(self._idx2word)))
        if len(l) + len(self._idx2word) < self.vocab_size:
            logging.warning('actual label set smaller than that configured: {}/{}'
                            .format(len(l) + len(self._idx2word), self.vocab_size))
        for word in ontology.all_domains_cross :#+ ['general']
            word = '[' + word + ']'
            self._add_to_vocab(word)
        for word in ontology.all_acts:#need to be updated for chinese version
            word = '[' + word + ']'
            self._add_to_vocab(word)
        for word in ontology.all_slots:#need to be updated for chinese version
            self._add_to_vocab(word)
        for word in l:
            if word.startswith('[value_') and word.endswith(']'):
                self._add_to_vocab(word)
        for word in l:
            self._add_to_vocab(word)
        self.vocab_size_oov = len(self._idx2word)
    def load_vocab(self, vocab_path):
        self._freq_dict = json.loads(open(vocab_path+'.freq.json', 'r').read().encode('utf-8'))
        self._word2idx = json.loads(open(vocab_path+'.word2idx.json', 'r').read().encode('utf-8'))
        self._idx2word = {}
        for w, idx in self._word2idx.items():
            self._idx2word[idx] = w
        self.vocab_size_oov = len(self._idx2word)
        print('vocab file loaded from "'+vocab_path+'"')
        print('Vocabulary size including oov: %d' % (self.vocab_size_oov))

    def save_vocab(self, vocab_path):
        _freq_dict = OrderedDict(sorted(self._freq_dict.items(), key=lambda kv:kv[1], reverse=True))
        write_dict(vocab_path+'.word2idx.json', self._word2idx)
        write_dict(vocab_path+'.freq.json', _freq_dict)
    
    def save_vocab_cross(self, vocab_path):
        _freq_dict = OrderedDict(sorted(self._freq_dict.items(), key=lambda kv:kv[1], reverse=True))
        write_dict_cross(vocab_path+'.word2idx.json', self._word2idx)
        write_dict_cross(vocab_path+'.freq.json', _freq_dict)


    def encode(self, word, include_oov=True):
        if include_oov:
            if self._word2idx.get(word, None) is None:
                raise ValueError('Unknown word: %s. Vocabulary should include oovs here.'%word)
            return self._word2idx[word]
        else:
            word = '<unk>' if word not in self._word2idx else word
            return self._word2idx[word]

    def sentence_encode(self, word_list):
        return [self.encode(_) for _ in word_list]

    def oov_idx_map(self, idx):
        return 2 if idx > self.vocab_size else idx

    def sentence_oov_map(self, index_list):
        return [self.oov_idx_map(_) for _ in index_list]


    def decode(self, idx, indicate_oov=False):
        if not self._idx2word.get(idx):
            raise ValueError('Error idx: %d. Vocabulary should include oovs here.'%idx)
        if not indicate_oov or idx<self.vocab_size:
            return self._idx2word[idx]
        else:
            return self._idx2word[idx]+'(o)'

    def sentence_decode(self, index_list, eos=None, indicate_oov=False):
        l = [self.decode(_, indicate_oov) for _ in index_list]
        if not eos or eos not in l:
            return ' '.join(l)
        else:
            idx = l.index(eos)
            return ' '.join(l[:idx])

    def nl_decode(self, l, eos=None):
        return [self.sentence_decode(_, eos) + '\n' for _ in l]

def padSeqs_gpt(sequences, pad_id, maxlen=None):
    lengths = []
    for x in sequences:
        lengths.append(len(x))

    num_samples = len(sequences)
    seq_mexlen = np.max(lengths)

    # maxlen = 1024
    if seq_mexlen > 1024: # gpt2.n_ctx
        # print('maxlen exceeds 1024')
        maxlen = 1024
    else:
        maxlen = seq_mexlen

    # tokenizer.encode('<|endoftext|>') = ['50256']
    # All labels set to ``-100`` are ignored (masked), the loss is only
    # computed for labels in ``[0, ..., config.vocab_size]`` (from modeling_gpt2.GPT2LMHeadModel)
    
    x = (np.ones((num_samples, maxlen)) * pad_id)
    for idx, s in enumerate(sequences):
        if not len(s):
            print('empty list was found in padSeqs')
        # trunc method = 'pre'
        trunc = s[-maxlen:]
        trunc = np.asarray(trunc)

        # pad method = 'post'
        x[idx, :len(trunc)] = trunc
            
    return x, lengths


    
        


def padSeqs(sequences, maxlen=None, truncated = False, pad_method='post',
                     trunc_method='pre', dtype='int32', value=0.):
    if not hasattr(sequences, '__len__'): 
        raise ValueError('`sequences` must be iterable.')
    lengths = []
    for x in sequences:
        if not hasattr(x, '__len__'):
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))
        lengths.append(len(x))

    num_samples = len(sequences)
    seq_maxlen = np.max(lengths)

    if maxlen is not None and truncated:
        maxlen = min(seq_maxlen, maxlen)
    else:
        maxlen = seq_maxlen
    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((num_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            print('empty list/array was found')
            continue  # empty list/array was found
        if trunc_method == 'pre':
            trunc = s[-maxlen:]
        elif trunc_method == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % trunc_method)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if pad_method == 'post':
            x[idx, :len(trunc)] = trunc
        elif pad_method == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % pad_method)
    return x


def get_glove_matrix(glove_path, vocab, initial_embedding_np):
    """
    return a glove embedding matrix
    :param self:
    :param glove_file:
    :param initial_embedding_np:
    :return: np array of [V,E]
    """
    ef = open(glove_path, 'r', encoding='UTF-8')
    cnt = 0
    vec_array = initial_embedding_np
    old_avg = np.average(vec_array)
    old_std = np.std(vec_array)
    vec_array = vec_array.astype(np.float32)
    new_avg, new_std = 0, 0

    for line in ef.readlines():
        line = line.strip().split(' ')
        word, vec = line[0], line[1:]
        vec = np.array(vec, np.float32)
        if not vocab.has_word(word):
            continue
        word_idx = vocab.encode(word)
        if word_idx <vocab.vocab_size:
            cnt += 1
            vec_array[word_idx] = vec
            new_avg += np.average(vec)
            new_std += np.std(vec)
    new_avg /= cnt
    new_std /= cnt
    ef.close()
    logging.info('%d known embedding. old mean: %f new mean %f, old std %f new std %f' % (cnt, old_avg,
                                                                                          new_avg, old_std, new_std))
    return vec_array


def position_encoding_init(self, n_position, d_pos_vec):
    position_enc = np.array([[pos / np.power(10000, 2 * (j // 2) / d_pos_vec) for j in range(d_pos_vec)]
                             if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1
    return position_enc

if __name__=='__main__':
    generate_value_set()
    get_data_related()
