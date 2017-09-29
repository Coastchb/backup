#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 10:25:49 2017

@author: coast
"""

'''
Extract MFCCs feature and generate phone-level labels. 
'''

import os
import sys
import pp

def sym2int(syms, sym_dict):
	return [sym_dict[sym] for sym in syms];

def process_raw(wav_file, config_file, feat_dir, label_dir, lexicon_dict):
    file_name = wav_file.split("/")[-1][ : -4];
    feat_file = os.path.join(feat_dir, "%s.mfcc" % file_name);
    
    ###extract MFCCs
    os.system("SMILExtract -C %s -I %s -O %s" % (config_file, wav_file, feat_file));
    
    ###generate the phone-level labels for each utterance
    raw_label_file = "%s.lab" % wav_file[:-4];
    phone_labels = [];
    raw_labels = open(raw_label_file).readlines()[0].strip().split();
    for raw_label in raw_labels:
        phone_labels.extend(lexicon_dict[raw_label]);
        
    phone_label_file = os.path.join(label_dir, "%s.lab" % file_name);
    with open(phone_label_file, "w") as fd:
        fd.writelines(" ".join(map(str, phone_labels)));    
    
def process_raw_batch(batch_dir, job_id, config_file, feat_dir, label_dir, lexicon_dict):
    wav_files = [file.strip() for file in open("%s/batch.%s.txt" % (batch_dir, job_id)).readlines()];
          
    job_server = pp.server(ppservers=());
    
    jobs = [job_server.submit(process_raw, (wav_file, config_file, feat_dir, 
                                            label_dir, lexicon_dict,),(),("os",)) for wav_file in wav_files];
    for job in jobs:
        job();
                
        
def main():
    batch_dir = sys.argv[1];
    job_id = sys.argv[2];
    target_feat_dir = sys.argv[3];
    target_lab_dir = sys.argv[4];
    config_file = sys.argv[5];
    phone_file = sys.argv[6];
    lexicon_file = sys.argv[7];
    
    phones = [line.strip() for line in open(phone_file).readlines()];
    phone_dict = dict(zip(phones, range(len(phones))));
    
    lexicon = [line.strip().split() for line in open(lexicon_file).readlines()];
    lexicon_dict = dict(zip([tokens[0] for tokens in lexicon], [sym2int(tokens[1 : ], phone_dict) for tokens in lexicon]));
  
    process_raw_batch(batch_dir, job_id, config_file, target_feat_dir, target_lab_dir, lexicon_dict);

main();