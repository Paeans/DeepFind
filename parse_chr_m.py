#!/usr/bin/env python

import os
import sys
import json
import numpy as np
from multiprocessing import Pool
#import Queue, threading


label_dir = "../share/label"
flabel_fname = "gz-file-list.txt"
slabel_suf = "-seg-label.json"
flabel_suf = "fea-label.json"
clabel_suf = "-label.txt"
segset_suf = "-seg.json"
segraw_suf = "-raw.json"
encode_suf = "-encode.json"

proc_num = 24
thrd_num = 10

seg_len = 200

raw_label_dict = {}

'''
#not used in this implemetation
#add multithreading
def create_point_label(q, fname, seg_id):
  if not fname in raw_label_dict: return  
  if seg_id in raw_label_dict[fname]:    
    q.put((fname, 1))
'''

'''
#not effective function

def create_seg_label(seg_id):
  flabel_list = json.load(open(label_dir + "/" + flabel_suf, 'r'))
  flabel_keys = flabel_list.keys()
  label = [0] * len(flabel_list)
  q = Queue.Queue()
  index = 0
  while index < len(flabel_list):
    if index + thrd_num < len(flabel_list):
      end = index + thrd_num
    else:
      end = len(flabel_list)
    tList = [threading.Thread(target=create_point_label, args = (q, x, seg_id))
                         for x in flabel_keys[index:end]]
    index += thrd_num
    [t.start() for t in tList]
    [t.join() for t in tList]
    while q.qsize() > 0:
      fname, tag = q.get()
      label[flabel_list[fname]] = tag
  return (seg_id, label)


#bad message length
while len(label_list) > 1:
if len(label_list)/proc_num >= 2:
  task_num = proc_num
else:
  task_num = len(label_list)/2
task_list = [label_list[i : i + len(label_list)/task_num] 
              for i in np.arange(task_num) * (len(label_list)/task_num)] 
for i in range(len(label_list) % task_num):
  task_list[i].append(label_list[-i-1])

label_list = p.map(merge_labels, task_list)
clabel_list = label_list[0]
'''

def dump_fea_label():
  index = 0
  flabel_list = {}  
  for line in [line.strip(' \n\t') for line in open(flabel_fname, 'r')]:
    flabel_list[line] = index
    index += 1
  json.dump(flabel_list, open(label_dir + "/" + flabel_suf, 'w'))
  return flabel_list

  
def dump_seg_label(clabel):
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
  json.dump(seg_set, open(label_dir + "/" + clabel + segset_suf, 'w'))
  json.dump(raw_label_dict, open(label_dir + "/" + clabel + segraw_suf, 'w'))
  return seg_set, raw_label_dict
  

def load_seg_label(clabel):
  seg_set_file = label_dir + "/" + clabel + segset_suf
  if os.path.isfile(seg_set_file):
    seg_set = json.load(open(seg_set_file, 'r'))
  else:
    seg_set = dump_seg_label(clabel)
  return seg_set
  
def dump_clabel_list(clabel):
  if not os.path.isfile(label_dir + "/" + flabel_suf):
    flabel_list = dump_fea_label()
  else:
    flabel_list = json.load(open(label_dir + "/" + flabel_suf, 'r'))
  
  if os.path.isfile(label_dir + "/" + clabel + segset_suf) and \
     os.path.isfile(label_dir + "/" + clabel + segraw_suf):
    seg_set = json.load(open(label_dir + "/" + clabel + segset_suf, 'r'))
    raw_label_dict = json.load(open(label_dir + "/" + clabel + segraw_suf, 'r'))
  else:
    seg_set, _ = dump_seg_label(clabel)
  
  fname_list = flabel_list.keys()
  fname_len = len(flabel_list)
  
  task_fname = [fname_list[i : i + fname_len/proc_num] 
                    for i in np.arange(proc_num) * (fname_len/proc_num)] 
  for i in range(fname_len % proc_num):
    task_fname[i].append(fname_list[-i-1])
  
  p = Pool(proc_num)
  label_list = p.map(create_seg_label, task_fname)
  clabel_list = {}
  
  for label in label_list:
    for seg_id in label:
      if seg_id in clabel_list:
        clabel_list[seg_id] = clabel_list[seg_id] + label[seg_id]
      else:
        clabel_list[seg_id] = label[seg_id]
  json.dump({x:clabel_list[x].tolist() for x in clabel_list}, 
               open(label_dir + "/" + clabel + slabel_suf, 'w'))
  

def create_seg_label(fnames):
  flabel_list = json.load(open(label_dir + "/" + flabel_suf, 'r'))
  label = {}
  flabel_num = len(flabel_list)
  for fname in fnames:
    if fname not in flabel_list or fname not in raw_label_dict: continue
    index = flabel_list[fname]
    
    seg_list = raw_label_dict[fname]
    for seg in seg_list:
      if seg in label:
        label[seg][index] = 1
      else:
        tLabel = np.zeros(flabel_num, dtype = np.int)
        tLabel[index] = 1
        label[seg] = tLabel
  return label

  
def merge_labels(label_list):
  label = {}
  for seg_dict in label_list:
    for seg_id in seg_dict:
      if seg_id in label:
        label[seg_id] = label[seg_id] + seg_dict[seg_id]
      else:
        label[seg_id] = seg_dict[seg_id]
  return label

  
if __name__ == "__main__":
  '''
  
  '''
  
  clabel = sys.argv[1]
  dump_clabel_list(clabel)
  


  
