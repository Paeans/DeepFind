from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import h5py
import argparse

import numpy as np
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.layers import LSTM

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adagrad, RMSprop, SGD

from keras.backend import tensorflow_backend as tback
from keras.utils import multi_gpu_model
from keras.regularizers import l1, l2, l1_l2

FLAGS = None

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
tback.set_session(tf.Session(config = config))

#gpu_index = '/device:GPU:1'

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
out_unit_1 = 1561
out_unit = 919
sample_shape = (25, 40)
chanel_size = 3

datafile = h5py.File('/home/pangaofeng/share/label/train.mat', 'r');
validfile = h5py.File('/home/pangaofeng/share/label/validn.mat', 'r');

data = datafile['trainxdata'];
label = datafile['traindata'];
valid = validfile['validxdata'];
lalid = validfile['validdata'];
print(data.shape, valid.shape)

def datagenerator(batch_size = 10):
  steps = data.shape[2]//batch_size;
  if data.shape[2] % batch_size != 0:
    steps += 1
  for i in range(steps):
    start = i * batch_size
    end = (i + 1) * batch_size
    res = np.moveaxis(data[:,:, start : end], -1, 0)
    yield res.reshape(res.shape[0], 125, 8 * res.shape[2]), \
          np.moveaxis(label[:, start : end], -1, 0)

def dg4fit(batch_size = 10):
  dg = datagenerator(batch_size = batch_size)
  while True:    
    try:
      yield dg.next()
    except StopIteration:
      dg = datagenerator(batch_size = batch_size)

def validgenerator(batch_size = 10):
  steps = valid.shape[2]//batch_size;
  if valid.shape[2] % batch_size != 0:
    steps += 1
  for i in range(steps):
    start = i * batch_size
    end = (i + 1) * batch_size
    res = np.moveaxis(valid[:,:, start : end], -1, 0)
    yield res.reshape(res.shape[0], 125, 8 * res.shape[2]), \
          np.moveaxis(lalid[:, start : end], -1, 0)
          
def dg4vl(batch_size = 10):
  vg = validgenerator(batch_size = batch_size)
  while True:
    try:
      yield vg.next()
    except StopIteration:
      vg = validgenerator(batch_size = batch_size)
      
def train_model():
  h, w = sample_shape
  c = chanel_size
  with tf.device('/device:GPU:' + str(FLAGS.gpu_index)):
    model = Sequential()
    model.add(LSTM(64, dropout=0.4, recurrent_dropout=0.4, 
                  kernel_regularizer=l1_l2(1e-8, 5e-7),
                  return_sequences = True, input_shape=(125, 32)))
    model.add(LSTM(128, dropout=0.4, recurrent_dropout=0.4, 
                  kernel_regularizer=l1_l2(1e-8, 5e-7),
                  return_sequences = True))
    model.add(LSTM(64, kernel_regularizer=l1_l2(1e-8, 5e-7)))
    #model.add(Dense(out_unit_1))
    model.add(Dense(out_unit, activation = 'sigmoid'))
  
  #multi_model = multi_gpu_model(model, gpus = 2)  
  #opt = Adagrad(lr = FLAGS.learning_rate, decay = 1e-6)
  #opt = RMSprop(lr = FLAGS.learning_rate, decay = 1e-3)
  #opt = SGD(lr = FLAGS.learning_rate, momentum = 0.01, decay = 1e-5)
  opt = Adagrad(lr = FLAGS.learning_rate, decay = 1e-5)
  model.compile(optimizer = opt, loss = 'binary_crossentropy', 
                  metrics = ['accuracy'])
  
  early_stopping = EarlyStopping(monitor='loss', patience=2)
  early_stopping = EarlyStopping(monitor='val_loss', patience=2)
  model_checkpoint = ModelCheckpoint(FLAGS.model_filename)
  
  steps_per_epoch = data.shape[2]//FLAGS.batch_size
  if data.shape[2]%FLAGS.batch_size != 0: steps_per_epoch += 1;
  validation_steps = valid.shape[2]//FLAGS.batch_size
  if valid.shape[2]%FLAGS.batch_size != 0: validation_steps += 1;
  with tf.device('/device:GPU:' + str(FLAGS.gpu_index)):
    model.fit_generator(generator = dg4fit(batch_size = FLAGS.batch_size),
                      steps_per_epoch = steps_per_epoch,
                      epochs = FLAGS.num_epochs,
                      validation_data = dg4vl(batch_size = FLAGS.batch_size),
                      validation_steps = validation_steps,
                      callbacks = [early_stopping, model_checkpoint])
    
  
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument("--batch_size", type=int, 
                        default=64, help="Size of batch fit to model")
  parser.add_argument("--num_epochs", type=int, 
                        default=100, help="Number of epochs to fit the model")
  parser.add_argument("--learning_rate", type=float, 
                        default=0.001, help="Learning rate")
  parser.add_argument("--train_algm", type=str, 
                        default='Adagrad', 
                        help="Algorithms used to train")
  parser.add_argument("--work_type", type=str, 
                        default="train", 
                        help="Type of operation: train, eval or prid")
  parser.add_argument("--model_filename", type=str, 
                        default="MY_MODEL_RNN_GPU1_ID000_{epoch:03d}-{loss:.2f}.HDF5", 
                        help="File name to store model")
  parser.add_argument("--gpu_index", type=int, 
                        default=1,
                        help="Indicator of use gpu")
  FLAGS, unparsed = parser.parse_known_args();
  train_model()
  