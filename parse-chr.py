#!/usr/bin/env python

import sys
import json
#from Bio.Seq import Seq

label_dir = "../share/label"
flabel_fname = "gz-file-list.txt"
slabel_suf = "-seg-label.json"
flabel_suf = "-fea-label.json"
clabel_suf = "-label.txt"

def index_flabel(file_name = flabel_fname):
  with open(file_name, 'r') as flabel_file:
    flabel_list = flabel_file.readlines()
  flabel_list = [x.strip() ]















if __name__ == "__main__":
  '''
  
  '''
  
  clabel = sys.argv[1]
  
  index = 0
  flabel_list = {}  
  for line in [line.strip(' \n\t') for line in open(flabel_fname, 'r')]:
    flabel_list[line] = index
    index += 1
  json.dump(flabel_list, open(label_dir + "/" + clabel + flabel_suf, 'w'))
  
  clabel_fname = label_dir + "/" + clabel + clabel_suf
  
  raw_label_dict = {}  
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
  
  counter = 0
  clabel_list = {seg_id:[0]*index for seg_id in seg_set}
  for seg_id in seg_set:
    for fname in raw_label_dict.keys():
      index = flabel_list[fname]
      seg_list = raw_label_dict[fname]
      while seg_id in seg_list:
        clabel_list[seg_id][index] = 1
        seg_list.remove(seg_id)
    counter += 1
    if counter % 1000 == 0: 
      with open(label_dir + "/" + clabel + "-log.txt", 'a') as logfile: 
        logfile.write(str(seg_id) + "\n")
  json.dump(clabel_list, open(label_dir + "/" + clabel + slabel_suf, 'w'))
  
  