import os

raw_mfcc = 'mfcc.txt';
tar_dir = './';


def split_mfcc(raw_mfcc):
    content = open(raw_mfcc).readlines();

    utt_mfcc = [];
    utt_id = "";

    for line in content:
        if(line.strartswith('s')):
            utt_id = line.strip()[:-1].strip();
        elif(']' in line):
            with open('%s/%s.txt' % (tar_dir, utt_id), "w") as fd:
                fd.writelines(''.join(utt_mfcc));
        else:
            utt_mfcc.append(line);

split_mfcc(raw_mfcc);		
