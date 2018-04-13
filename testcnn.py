from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import h5py
import argparse
import hdf5storage as hf

import numpy as np
import tensorflow as tf

from sklearn import metrics

from keras.models import load_model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adagrad

from keras.backend import tensorflow_backend as tback
from keras.utils import multi_gpu_model

FLAGS = None

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
tback.set_session(tf.Session(config = config))

colorindex = np.array(
                [[0, 0, 255], 
                 [0, 255, 255], 
                 [255, 0, 0], 
                 [255, 255, 0]], 
                dtype = np.float32)
                
kernel_num = [320, 480, 960]
kernel_size = [(2,3), (2,3), (2,3)]
pool_size = [(2,2), (2,2)]
drop_rate = [0.2, 0.2, 0.5]
out_unit = 919
sample_shape = (25, 40)
chanel_size = 3

datafile = h5py.File('/home/pangaofeng/share/label/testn.mat', 'r');
data = datafile['testxdata'];
label = datafile['testdata'];
print(data.shape)

with h5py.File('oauc.mat', 'r') as oauc_file:  
  oauc = np.array(oauc_file['oauc'], dtype=np.float32).T

def datagenerator(batch_size = 10):
  steps = data.shape[2]//batch_size;
  if data.shape[2] % batch_size != 0:
    steps += 1
  for i in range(steps):
    start = i * batch_size
    end = (i + 1) * batch_size
    res = np.matmul(np.moveaxis(data[:,:, start : end], -1, 0), colorindex)
    yield res.reshape(res.shape[0], 25, 40, res.shape[2]), \
          np.moveaxis(label[:, start : end], -1, 0)

def dg4pred(batch_size = 10):
  dg = datagenerator(batch_size = batch_size)
  while True:    
    try:
      d, l = dg.next()
      yield d
    except StopIteration:
      dg = datagenerator(batch_size = batch_size)
    

def test_model():
  sample_size = data.shape[2]
  steps = sample_size // FLAGS.batch_size
  if sample_size % FLAGS.batch_size != 0:
    steps += 1
  
  with tf.device('/device:GPU:' + str(FLAGS.gpu_index)):
    model = load_model(FLAGS.model_filename, compile = False)
    result = model.predict_generator(
          generator = dg4pred(batch_size = FLAGS.batch_size), 
          steps = steps)
  print(result.shape)
  auc_result = []  
  for i in range(result.shape[1]):
    if max(label[i, :]) < 1: auc_score = 0.5
    else: auc_score = metrics.roc_auc_score(label[i, :], result[:, i])
    auc_result.append(auc_score)
    
  mauc = np.array(auc_result, dtype=np.float32)
  mat_file_name = '_'.join(re.split('_|-', FLAGS.model_filename)[2:6])
  hf.savemat(mat_file_name.upper() + '_auc.mat', 
              {'auc_result':mauc}, format='7.3')
              
  print('The mean and std of OAUC: %.4f %.4f' % (np.mean(oauc), np.std(oauc)))
  print('The mean and std of MAUC: %.4f %.4f %.4f %.4f' % (np.mean(mauc), np.std(mauc), 
                                                    np.max(mauc), np.min(mauc)))
  print('AUC great than other is: %d' % np.sum(mauc >= oauc))
 
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument("--batch_size", type=int, 
                      default=64, help="Size of batch fit to model")
  parser.add_argument("--num_epochs", type=int, 
                      default=5, help="Number of epochs to fit the model")
  parser.add_argument("--learning_rate", type=float, 
                      default=0.01, help="Learning rate")
  parser.add_argument("--train_algm", type=str, 
                        default='Adagrad', 
                        help="Algorithms used to train")
  parser.add_argument("--work_type", type=str, 
                        default="train", 
                        help="Type of operation: train, eval or prid")
  parser.add_argument("--model_filename", type=str, 
                        default=None, 
                        help="File name to store model")
  parser.add_argument("--use_gpu", type=bool, default="true",
                        help="Flag to use gpu or not")
  parser.add_argument("--gpu_index", type=int, 
                        default=1,
                        help="Indicator of use gpu")
  FLAGS, unparsed = parser.parse_known_args();
  test_model()
  
