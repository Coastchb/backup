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
import glob
#import subprocess

def sym2int(syms, sym_dict):
	return [sym_dict[sym] for sym in syms];

def processRaw(raw_dir, config_file, feat_dir, label_dir, lexicon_dict):
    wav_files = glob.glob("%s/*.wav" % raw_dir);
          
    for wav_file in wav_files:
        file_name = wav_file.split("/")[-1][ : -4];
        feat_file = os.path.join(feat_dir, "%s.mfcc" % file_name);
        phone_label_file = os.path.join(label_dir, "%s.lab" % file_name);
        
        ###extract MFCCs
        os.system("SMILExtract -C %s -I %s -O %s" % (config_file, wav_file, feat_file));
            
        ###generate the phone-level labels for each utterance
        raw_label_file = os.path.join("%s/%s.lab" % (raw_dir, file_name));
        phone_labels = [];
        raw_labels = open(raw_label_file).readlines()[0].strip().split();
        for raw_label in raw_labels:
            phone_labels.extend(lexicon_dict[raw_label]);

        with open(phone_label_file, "w") as fd:
                fd.writelines(" ".join(map(str, phone_labels)));
                
        
def main():
    raw_dir = "data/train/raw" #"/home/dear/archives/data/digits/train";
    feat_dir = "data/train/feat";
    label_dir = "data/train/phn_lab";
    config_file = "config/MFCC39_E_D_A_Z.conf";
    phone_file = "data/lm/nonsilence_phones.txt";
    lexicon_file = "data/lm/nonsilence_lexicon.txt";
    
    if(not (os.path.exists(feat_dir) and os.path.exists(label_dir))):
        os.makedirs(feat_dir);
        os.makedirs(label_dir);                 
        
    phones = [line.strip() for line in open(phone_file).readlines()];
    phone_dict = dict(zip(phones, range(len(phones))));
    
    lexicon = [line.strip().split() for line in open(lexicon_file).readlines()];
    lexicon_dict = dict(zip([tokens[0] for tokens in lexicon], [sym2int(tokens[1 : ], phone_dict) for tokens in lexicon]));
  
    processRaw(raw_dir, config_file, feat_dir, label_dir, lexicon_dict);

main();