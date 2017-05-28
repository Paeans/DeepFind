#!/usr/bin/env python

import sys
import json
from multiprocessing import Pool
import Queue, threading
#from Bio.Seq import Seq

label_dir = "../share/label"
flabel_fname = "gz-file-list.txt"
slabel_suf = "-seg-label.json"
flabel_suf = "fea-label.json"
clabel_suf = "-label.txt"

proc_num = 24
thrd_num = 10

raw_label_dict = {}

def creat_seg_label(seg_id):
  flabel_list = json.load(open(label_dir + "/" + flabel_suf, 'r'))
  
  label = [0] * len(flabel_list)
  for fname in flabel_list:
    if not fname in raw_label_dict: continue
    
    if seg_id in raw_label_dict[fname]:
      label[flabel_list[fname]] = 1
  return (seg_id, label)


if __name__ == "__main__":
  
  clabel = sys.argv[1]
  
  index = 0
  flabel_list = {}  
  for line in [line.strip(' \n\t') for line in open(flabel_fname, 'r')]:
    flabel_list[line] = index
    index += 1
  json.dump(flabel_list, open(label_dir + "/" + flabel_suf, 'w'))
  
  clabel_fname = label_dir + "/" + clabel + clabel_suf
  
  
  for line in open(clabel_fname, 'r'):
    seg_list = line.strip(' \n\t').split()
    fname = seg_list[0]
    if len(seg_list) > 1:
      seg_list = [int(x) for x in seg_list[1:]]
    else:
      seg_list = []
    seg_list.sort()
    raw_label_dict[fname] = seg_list
  seg_set = sorted(list(set().union(*raw_label_dict.values())))
  json.dump(flabel_list, open(label_dir + "/" + clabel + "-seg.json", 'w'))
  
  p = Pool(proc_num)
  label_list = p.map(creat_seg_label, seg_set)
  
  clabel_list = {seg_id:label for seg_id, label in label_list}
  json.dump(clabel_list, open(label_dir + "/" + clabel + slabel_suf, 'w'))
  
  
