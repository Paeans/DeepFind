from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import gzip
import argparse

from Bio import SeqIO

from multiprocessing import Pool

peak_data_dir = '../share/narrowPeak'
hg_data_dir   = '../share/hg19'
label_data_dir = '../share/label'
feature_list_file_name = 'gz-file-list.txt'

proc_num = 24
segment_len = 200

feature_length = 918

def create_feature_list():
  feature_list = {}
  index = 0
  with open(feature_list_file_name, 'r') as fl:
    for l in fl:
      feature_list[l.strip()] = index
      index += 1
  return index, feature_list


def create_peak_dict(peak_file_name):
  result = {}
  if not os.path.isfile(peak_data_dir + '/' + peak_file_name):
    print('ERROR:', peak_file_name, 'not exist in', peak_data_dir)
    return result
  
  with gzip.open(peak_data_dir + '/' + peak_file_name, 'r') as pfile:
    for line in pfile:
      chrname, start_bp, end_bp = line.strip().split()[0:3]
      if chrname == 'chrX' or chrname == 'chrY' or chrname == 'chrM': continue
      try:
        ##seg_index = (int(start_bp) + int(end_bp)) // 2 // segment_len
        
        cal_index_fun = lambda x: int(x) // segment_len + \
                             (0 if int(x) % segment_len <= segment_len // 2
                                else 1)
        start_index = cal_index_fun(start_bp)
        end_index = cal_index_fun(end_bp)
        
        '''
        if chrname in result:
          result[chrname].append(seg_index)
        else:
          result[chrname] = [seg_index]
        '''
        
        if chrname not in result:
          result[chrname] = set()
        ##result[chrname].add(seg_index)
        
        [result[chrname].add(x) for x in range(start_index, end_index)]
        
      except ValueError:
        print('ERROR: not integer value occured in', peak_file_name)
        print('ERROR: the line is', line)
        continue
  result['name'] = peak_file_name
  return result


def create_seq_label_map(peak_list_dict):
  peak_set = set()
  peak_label_dict = {}
  result = []
  
  chrname = peak_list_dict['name']
  
  if not os.path.isfile(hg_data_dir + '/' + chrname + '.fa'):
    print('ERROR:', chrname, 'not exist in', hg_data_dir)
    return result
  
  for feature_index in peak_list_dict.keys():
    if feature_index == 'name': continue
    
    peak_list = peak_list_dict[feature_index]
    for peak in peak_list:
      peak_set.add(peak)
      
      if peak not in peak_label_dict:
        peak_label_dict[peak] = [0 for x in range(feature_length)]      
      peak_label_dict[peak][feature_index] = 1
  
  peak_list = sorted(list(peak_set))
  for record in SeqIO.parse(hg_data_dir + '/' + chrname + '.fa', "fasta"):
    cname = record.id
    if not cname == chrname: continue
    for peak in peak_list:
      segment = record[(peak - 2) * segment_len : (peak + 3) * segment_len].upper()
      result.append((segment, peak_label_dict[peak]))
      
  return result


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--data_type", 
      type=str, 
      default='train', 
      help="Type of data need to generate")
  FLAGS, unparsed = parser.parse_known_args()
  
  feature_length, feature_list = create_feature_list()
  
  p = Pool(proc_num)
  result_list = p.map(create_peak_dict, feature_list.keys())
  
  chrname_peak_dict = {}
  for result in result_list:
    peak_name_index = feature_list[result['name']]
    for chrname in result.keys():
      if chrname == 'name': continue
      if (FLAGS.data_type == 'train' and chrname == 'chr22') or \
          (FLAGS.data_type == 'test' and not chrname == 'chr22'): continue
      if chrname not in chrname_peak_dict:
        chrname_peak_dict[chrname] = {'name':chrname}
      chrname_peak_dict[chrname][peak_name_index] = result[chrname]
  
  chrname_peak_dict_len = len(chrname_peak_dict)
  '''
  # multiprocessing is not effective, Bio read files, then Disk load is high
  seq_label_list = Pool(chrname_peak_dict_len).map(create_seq_label_map, chrname_peak_dict.values())
  '''
  
  if FLAGS.data_type == 'train':
    result_name = label_data_dir + '/' + 'train_data.gz' 
  elif FLAGS.data_type == 'test':
    result_name = label_data_dir + '/' + 'test_data.gz' 
  
  counter = 0
  with gzip.open(result_name, 'w') as tdfile:
    for peak_list_dict in chrname_peak_dict.values():
      turple_list = create_seq_label_map(peak_list_dict)
      
      #tdfile.write('\n'.join([str(segment.seq) + ' ' + ''.join(str(x) for x in label) for segment, label in turple_list]))
      #counter += len(turple_list)
      for segment, label in turple_list:
        tdfile.write(str(segment.seq) + ' ' + ''.join(str(x) for x in label))        
        tdfile.write('\n')
        counter += 1
  print(counter)
  