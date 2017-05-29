#!/usr/bin/env python

import os
import sys
import json
import numpy as np
from Bio import SeqIO
from parse_chr_m import *

fa_suf = ".fa"
hg_dir = "../share/hg19"

ntype_value  = 0.25
sta_offset = -2
end_offset   = 2

def encode_bp(bp):
  if bp == 'N': 
    return list(np.full(4, ntype_value, dtype = np.int))
  vDict = {'A':0, 'T':1, 'C':2, 'G':3}
  bp_coder = np.zeros(4, dtype = np.int)
  bp_coder[vDict[bp]] = 1
  return list(bp_coder)


  
if __name__ == "__main__":
  fa_file = sys.argv[1]
   
  if not os.path.isfile(fa_file):
    print "No fasta file " + fa_file + " in " + hg_dir
    exit(1)
    
  for record in SeqIO.parse(fa_file, "fasta"):
    cname = record.id
    clen  = len(record)
    encode_dict = {}
    
    seg_set = load_seg_label(cname)
    for seg_id in seg_set:
      stabp = (seg_id + sta_offset) * seg_len
      endbp = (seg_id + end_offset) * seg_len
      if stabp < 0 or endbp > clen: continue
      encode_dict[seg_id] = [encode_bp(x) 
        for x in record[stabp : endbp].upper()]
    json.dump(encode_dict, open(label_dir + "/" + cname + encode_suf, 'w'))
  
  
