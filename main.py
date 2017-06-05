#!/usr/bin/env python
import time
import os
import sys
import json
import numpy as np
from Bio import SeqIO
from parse_chr_m import *

fa_suf = ".fa"
hg_dir = "../share/hg19"

ntype_value  = 0    #.25
sta_offset   = -2
end_offset   = 3

def encode_bp(bp):
  if bp == 'N': 
    return np.full((4,1), ntype_value, dtype = np.int)
  vDict = {'A':0, 'T':1, 'C':2, 'G':3}
  bp_coder = np.zeros((4,1), dtype = np.int)
  bp_coder[vDict[bp]][0] = 1
  return bp_coder

#multiprocessing not effective
#disk have very huge load
def encode_seg(seg_dict):
  result_dict = {}
  for seg_id in seg_dict.keys():
    tmp = np.zeros((4,0), dtype=np.int)
    for x in seg_dict[seg_id]:
      tmp = np.append(tmp, encode_bp(x), axis = 1)
    result_dict[seg_id] = tmp
  return result_dict

def map_id_seg(seg_list, seg_dict):
  result = {}
  for x in seg_list:
    if not x in seg_dict: continue
    result[x] = seg_dict[x]
  return result

def devide_seg_set(seg_set, seg_dict):
  seg_len = len(seg_set)
  task_sname = [seg_set[i : i + seg_len/proc_num] 
                    for i in np.arange(proc_num) * (seg_len/proc_num)] 
  for i in range(seg_len % proc_num):
    task_sname[i].append(seg_set[-i-1])
  return [map_id_seg(x, seg_dict) for x in task_sname]

  
if __name__ == "__main__":
  fa_file = sys.argv[1]
  start_time = time.time() 
  if not os.path.isfile(fa_file):
    print "No fasta file " + fa_file + " in " + hg_dir
    exit(1)
  
  '''
  with open(fa_file, 'r') as fa:
    cname_list = [record.id for record in SeqIO.parse(fa, "fasta")]
  p = Pool(proc_num)
  for cname in cname_list:
    encode_dict = {}
    
    seg_set = load_seg_label(cname)
    encode_list = p.map(encode_seg, 
      [{'fa_file':fa_file, 'cname':cname, 'task_list':x} 
        for x in devide_seg_set(seg_set)])
    print time.time() - start_time
    for dencode in encode_list:
      for x in dencode:
        encode_dict[x] = dencode[x]
    print time.time() - start_time
    #json.dump(encode_dict, open(label_dir + "/" + cname + encode_suf, 'w'))
  '''
  
  for record in SeqIO.parse(fa_file, "fasta"):
    cname = record.id
    clen  = len(record)
    encode_dict = {}
    seg_dict = {}
    seg_set = load_seg_label(cname)
    for seg_id in seg_set:
      stabp = (seg_id + sta_offset) * seg_len
      endbp = (seg_id + end_offset) * seg_len
      if stabp < 0 or endbp > clen: continue
      seg_dict[seg_id] = record[stabp : endbp].upper()
      
    p = Pool(proc_num)
    encode_list = p.map(encode_seg, devide_seg_set(seg_set, seg_dict))
    for r in encode_list:
      encode_dict.update(r)
    print time.time() - start_time
    result = {}
    for x in encode_dict.keys():
      result[x] = encode_dict[x].tolist()
    json.dump(result, open(label_dir + "/" + cname + encode_suf, 'w'))
  