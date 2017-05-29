#!/usr/bin/env python

import os
import sys
import json
from Bio import SeqIO
from parse_chr_m import *

fa_suf = ".fa"
hg_dir = "../share/hg19"

if __name__ == "__main__":
  fa_file = sys.argv[1]
  
  seg_set_file = label_dir + "/" + clabel + segset_suf
  if os.path.isfile(seg_set_file):
    seg_set = json.load(open(seg_set_file, 'r'))
  else:
    seg_set = dump_seg_label(clabel)
  
  #fa_file = hg_dir + "/" + clabel + fa_suf
  if not os.path.isfile(fa_file):
    print "No fasta file " + fa_file + " in " + hg_dir
    return 1
  
  
  encode_dict = {}
  