#%%
import numpy as np
import os
import random

#%%

def get_files(feat_dir, lab_dir, with_header, start_index):
    '''
    Get all the feat files and label files without shuffle.
    
    Args:
        feat_dir: feature directory
        lab_dir: label directory
        
    Returns:
        list of features and labels
    '''
    feat_file_paths = [];   #array, containing the feat file path to an uttearance       
    labels = [];   #1-D array, each element is the phone label to an utterance
    for feat_file in os.listdir(feat_dir):
        file_name = feat_file[ : -5];
        lab_file_path = os.path.join(lab_dir, "%s.lab" % file_name);
        
        if(not os.path.exists(lab_file_path)):
            print("Error: %s not exists." % lab_file_path);
            exit();
        
        feat_file_paths.append(os.path.join(feat_dir, feat_file));
        labels.append(open(lab_file_path).readlines()[0].strip());
        
    temp = [(feat_file_path, label) for feat_file_path, label in zip(feat_file_paths, labels)];
#    random.shuffle(temp)
    
    
    file_path_list = [obj[0] for obj in temp];
    label_list = [obj[1] for obj in temp];
#    label_list = [int(i) for i in label_list]
    
    feat_dim, max_time_step = get_feat_info(file_path_list, with_header, start_index);
    return file_path_list, label_list, feat_dim, max_time_step

#%%
def get_feat_info(feat_file_paths, with_header, start_index=5):
    feat_dim, max_time_step = 0, 0;
    
    for feat_file_path in feat_file_paths:
        feat_content = open(feat_file_path).readlines();
        
        if(feat_dim == 0):
            feat_dim = len(feat_content[0].split(',')) - start_index;
        utt_len = len(feat_content) if not with_header else len(feat_content) - 1;
        max_time_step = max(max_time_step, utt_len);
        
    return feat_dim, max_time_step;
        
#%%

def get_batchs(feat_file_paths, labels, feat_dim, batch_size):
    '''
    Shuffle all the data and then split to get batchs.
    Args:
        feat_file_paths: list of feat file paths
        labels: list of phone-level labels
#        batch_size: batch size
        capacity: the maximum elements in queue
    Returns:
        feat_batch: 3-D tensor [batch_size, width, height], dtype=tf.float32
        label_batch: 2-D tensor [batch_size, utt_len], dtype=tf.int32
    '''
    featFiles_labels = np.array([(featFile, label) for featFile,label in zip(feat_file_paths, labels)]);
    np.random.shuffle(featFiles_labels);
    
    total_sample = len(feat_file_paths);
    num_batch = int(total_sample / batch_size);

    batch_data_num = num_batch * batch_size;
    featFiles_labels = featFiles_labels[ : batch_data_num];
    sample_batchs = np.vsplit(featFiles_labels, num_batch);
    
    featFiles_batchs = [[sample[0] for sample in sample_batch] for sample_batch in sample_batchs];
    label_batchs = [[sample[1] for sample in sample_batch] for sample_batch in sample_batchs];
    
    return featFiles_batchs, label_batchs; 
    
#%%
def read_batch_data(file_batch, label_batch, max_time_step, with_header, start_index):	
    
    batch_feats, batch_labels = [], [];
    for feat_file, label in zip(file_batch, label_batch):
        feats = read_feat(feat_file, with_header, start_index);
        batch_feats.append(feats);
        batch_labels.append(list(map(int, label.split())));
        
    assert len(file_batch) == len(label_batch)
    feat_dim = batch_feats[0].shape[0]
        
    batch_size = len(file_batch);
    batch_inputs = np.zeros((max_time_step, batch_size, feat_dim))
    batch_seq_lens = []

    for sample_index in range(batch_size):
        feat = batch_feats[sample_index];
        pad_secs = max_time_step - feat.shape[1];
        batch_inputs[:,sample_index,:] = np.pad(feat.T, ((0,pad_secs),(0,0)),
                                             'constant', constant_values=0)  
        batch_seq_lens.append(batch_feats[sample_index].shape[1])

    return batch_inputs, target_list_to_sparse_tensor(batch_labels), batch_seq_lens; 

#%%
def read_feat(feat_file, with_header, start_index=5):
    feat_content = open(feat_file).readlines();
    if(with_header):
        feat_content = feat_content[1:]
    utt_feats = [list(map(float, line.strip().split(',')[start_index : ])) for line in feat_content];
    
    return np.array(utt_feats).transpose();


#%%
def target_list_to_sparse_tensor(targetList):
    '''make tensorflow SparseTensor from list of targets, with each element
       in the list being a list or array with the values of the target sequence
       (e.g., the integer values of a character map for an ASR target string)
       See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/ctc/ctc_loss_op_test.py
       for example of SparseTensor format'''
    indices = []
    vals = []
    for tI, target in enumerate(targetList):
        for seqI, val in enumerate(target):
            indices.append([tI, seqI])
            vals.append(val)
    shape = [len(targetList), np.asarray(indices).max(0)[1]+1]
    return (np.array(indices), np.array(vals), np.array(shape))
  
#%%
#for test
#    
#BATCH_SIZE = 2
#feat_dir = 'data/tmp/';
#lab_dir = 'data/train/phn_lab/';
#feat_file_list, label_list, feat_dim, max_time_step = get_files(feat_dir, lab_dir);
#
#featFiles_batchs, label_batchs = get_batchs(feat_file_list, label_list, feat_dim, max_time_step, BATCH_SIZE);
#
#print(featFiles_batchs);
#print(label_batchs);
