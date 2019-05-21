# -*- coding: utf-8 -*-

import os
import configparser 

config_file = "config.cfg";
raw_gram_file = "Raw.gram"; ### raw input

gram_file = "bank.gram"; 
fsg_file = "bank.fsg";  
fst_file = "bank.fst.txt";
fst_compiled_file =  "bank.fst";
syms_file = "bank.sym";
pdf_file = "bank.pdf";

raw_grams = open(raw_gram_file).readlines();
tar_grams = [];
config = configparser.ConfigParser();
config.read(config_file);
section = config['DEFAULT'];
for raw_gram in raw_grams:
    tar_gram = raw_gram;
    for key in section:
        key = key.upper();
        tar_gram = tar_gram.replace(key, section[key]);

    tar_grams.append(tar_gram);
    
with open(gram_file, "w") as fd:
    fd.writelines(''.join(tar_grams));

os.system("sphinx_jsgf2fsg -jsgf %s -fsg %s " % (gram_file, fsg_file));


fsg_content = open(fsg_file).readlines();

trans = [line.strip().replace("TRANSITION ","") for line in fsg_content if line.startswith("TRANSITION")];

trans_tokens = [tran.split() for tran in trans];

trans_tokens = [tran_tokens + ['<eps>'] if len(tran_tokens) < 4 else tran_tokens for tran_tokens in trans_tokens];

syms = list(set([tran_tokens[-1] for tran_tokens in trans_tokens]));

syms_num = len(syms);
syms_int = dict(zip(syms, range(syms_num)));

with open(syms_file, "w") as fd:
    for i in range(syms_num):
        fd.writelines("%s %d\n" % (syms[i], i));
    
spec_syms = [section[key] for key in section];    
with open(fst_file, "w") as fd:
    max_id = 0;
    for tran_tokens in trans_tokens:
#        print tran_tokens;
        sym = tran_tokens[-1];
        sym_id = syms_int[sym];
        fd.writelines("%s %s %d %d" % (tran_tokens[0], tran_tokens[1], syms_int['<eps>'] if sym in spec_syms else sym_id, sym_id));
        fd.writelines("\n");
        max_id = max(max_id, int(tran_tokens[1]));
    fd.writelines("%d\n" % max_id);

       
os.system("fstcompile %s | fstdeterminize > %s " % (fst_file, fst_compiled_file));

os.system("fstdraw --isymbols=%s --osymbols=%s %s " \
          "| dot -Tpdf -Gsize=8.5G  > %s" % (syms_file, syms_file, fst_compiled_file, pdf_file) );