#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 11:01:12 2017

@author: dear
"""

import sys
import glob

raw_dir = sys.argv[1];
num_job = int(sys.argv[2]);
batch_dir = sys.argv[3];

wav_files = glob.glob("%s/*.wav" % raw_dir);

total_num = len(wav_files);
start = 0;
batch_size = int(total_num / num_job) + 1;

for batch_index in range(num_job):
    end = start + batch_size;
    
    with open("%s/batch.%d.txt" % (batch_dir, batch_index), "w") as fd:
        fd.writelines('\n'.join(wav_files[start:end]));
    
    start = end;

