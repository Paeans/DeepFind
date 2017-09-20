from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse
import json

import h5py
import hdf5storage as hf
import numpy as np
import tensorflow as tf

from sklearn import metrics

from tensorflow.contrib import learn
from tensorflow.contrib import layers
from tensorflow.contrib.learn import io as numpy_io
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

FLAGS = None

kernel_num = [320, 480, 960]
out_unit = 919

segment_len = 1000
feature_len = 919

data_dir = '../label'
train_file = 'train.mat'
test_file = 'testn.mat'

#model_regular = layers.sum_regularizer(
#        [tf.contrib.layers.l1_regularizer(1e-08),
#         tf.contrib.layers.l2_regularizer(5e-07)])
model_regular = layers.sum_regularizer([tf.contrib.layers.l2_regularizer(5e-07)])

def cnn_model_fn(features, labels, mode, params):
  print(features, labels)  
  
  input_layer = tf.reshape(features['train_data'], [-1, 4, 1000, 1])
  
  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 4, 1000, 1]
  # Output Tensor Shape: [batch_size, 4, 1000, 32]
  conv1 = layers.conv2d(
      inputs=input_layer,
      num_outputs=kernel_num[0],
      kernel_size=[1, 8],
      stride=1,
      padding="VALID",
      activation_fn=tf.nn.relu,
      weights_regularizer=model_regular)
      #normalizer_fn=tf.contrib.keras.constraints.max_norm(max_value=0.9, axis=[0, 1, 2])
      
  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 1, 1000, 320]
  # Output Tensor Shape: [batch_size, 1, 250, 320]
  pool1 = layers.max_pool2d(inputs=conv1, kernel_size=[1, 4], stride=[1, 4], padding='VALID')
  level1 = layers.dropout(inputs=pool1, keep_prob=0.8)

  # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 1, 250, 32]
  # Output Tensor Shape: [batch_size, 1, 250, 64]
  conv2 = layers.conv2d(
      inputs=level1,
      num_outputs=kernel_num[1],
      kernel_size=[1, 8],
      stride=1,
      padding="VALID",
      activation_fn=tf.nn.relu,
      weights_regularizer=model_regular)

  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 1, 250, 64]
  # Output Tensor Shape: [batch_size, 1, 125, 64]
  pool2 = layers.max_pool2d(inputs=conv2, kernel_size=[1, 4], stride=[1, 4], padding='VALID')
  level2 = layers.dropout(inputs=pool2, keep_prob=0.8)

  conv3 = layers.conv2d(
      inputs=level2,
      num_outputs=kernel_num[2],
      kernel_size=[1, 8],
      stride=1,
      padding="VALID",
      activation_fn=tf.nn.relu,
      weights_regularizer=model_regular)

  level3 = layers.dropout(inputs=conv3, keep_prob=0.5)

  # Flatten tensor into a batch of vectors
  level3_flat = layers.flatten(level3)
  
  full_connect1 = layers.fully_connected(
      inputs=level3_flat,
      num_outputs=out_unit,
      weights_regularizer=model_regular
      )

  full_connect2 = layers.fully_connected(
      inputs=full_connect1,
      num_outputs=out_unit,
      activation_fn=None,
      weights_regularizer=tf.contrib.layers.l1_regularizer(1e-08))
      
  logits = tf.sigmoid(full_connect2, name='logits_results')
  
  loss = None
  train_op = None
  
  
  # Calculate Loss (for both TRAIN and EVAL modes)
  if mode != learn.ModeKeys.INFER:
    #onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    print(labels, logits)
    loss = tf.losses.sigmoid_cross_entropy(
        multi_class_labels=labels, logits=full_connect2)
    # loss = tf.contrib.keras.losses.binary_crossentropy(
        # y_true=labels, y_pred=logits)
  
  # Configure the Training Op (for TRAIN mode)
  if mode == learn.ModeKeys.TRAIN:
    global_step = tf.contrib.framework.get_global_step()
    learning_rate = tf.train.exponential_decay(params['learning_rate'], global_step,
                                           100, 0.99, staircase=True)
    train_op = layers.optimize_loss(
        loss=loss,
        global_step=global_step,
        learning_rate=learning_rate, #0.001,
        optimizer=FLAGS.train_algm) #"SGD""Adagrad"
    #"Adagrad": train.AdagradOptimizer,
    #"Adam": train.AdamOptimizer,
    #"Ftrl": train.FtrlOptimizer,
    #"Momentum": train.MomentumOptimizer,
    #"RMSProp": train.RMSPropOptimizer,
    #"SGD": train.GradientDescentOptimizer,

  # Generate Predictions
  
  predictions = logits
  #{
  #    "logitresult": logits
  #}#Generate predictions_key_list, 1:918

  # Return a ModelFnOps object
  return model_fn_lib.ModelFnOps(
      mode=mode, predictions=predictions, loss=loss, train_op=train_op)

def load_data_from_file(filename, startline, endline, multi = 1000):
  
  if not os.path.isfile(filename):
    print('ERROR:', filename, 'not exist')
    return None, None
  
  datafile = h5py.File(filename, 'r')
  h5_label = datafile['traindata']
  h5_data = datafile['trainxdata']
  
  target = h5_label[:,startline * multi : endline * multi]
  data = h5_data[:,:,startline * multi : endline * multi]
  datafile.close()
  
  return np.transpose(data, (2,1,0)).astype(dtype = np.float32), \
          np.transpose(target, (1,0)).astype(dtype = np.float32)
  
def load_test_from_file(filename, startline, endline, multi = 1000):
  
  if not os.path.isfile(filename):
    print('ERROR:', filename, 'not exist')
    return None, None
  
  datafile = h5py.File(filename, 'r')
  h5_label = datafile['testdata']
  h5_data = datafile['testxdata']
  
  target = h5_label[:,startline * multi : endline * multi]
  data = h5_data[:,:,startline * multi : endline * multi]
  datafile.close()
  return np.transpose(data, (2,1,0)).astype(dtype = np.float32), \
          np.transpose(target, (1,0)).astype(dtype = np.float32)
  
  
def main(unused_argv):
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  config=tf.contrib.learn.RunConfig(session_config=config)
  gene_classifier = learn.Estimator(
          model_fn=cnn_model_fn, 
          model_dir=FLAGS.model_dir,
          params={'learning_rate':FLAGS.learning_rate},
          config=config)
  
  train_data_file = data_dir + '/' + train_file
  startline = FLAGS.startline
  endline = FLAGS.endline
  load_size = 1000
  
  while True:
    if startline > endline: break
    train_data, train_labels = load_data_from_file(
                                   train_data_file, 
                                   startline = startline, 
                                   endline = startline + load_size)
    
    print(train_data.shape, train_labels.shape)
    if train_data.shape[0] == 0: break
    
    input_fn = numpy_io.numpy_input_fn(
        {'train_data':train_data}, 
        train_labels,
        batch_size=FLAGS.batch_size,
        shuffle=False,
        num_epochs=FLAGS.num_epochs)
    
    # Set up logging for predictions
    # Log the values in the "logits" tensor with label "logits_results"
    tensors_to_log = {"logitresult": "logits_results"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=10000)
    # Train the model
  
    gene_classifier.fit(
        input_fn=input_fn,
        #steps=FLAGS.steps_size,  #sizeofdata/batchsize * epoch
        monitors=[logging_hook])
    
    startline = startline + load_size
    print('Finish sub task training:', startline)
    sys.stdout.flush()
  print("Finish training")

def pred(unused_argv):
  print("Prediction")
  test_data_file = data_dir + '/' + test_file
  with tf.name_scope('predict_scope') as scope:
    with tf.device('/gpu:1'):
      train_data, train_labels = \
            load_test_from_file(
              test_data_file, 
              startline = FLAGS.startline, 
              endline = FLAGS.endline)
                                    
      config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
      config.gpu_options.allow_growth = True
      config=tf.contrib.learn.RunConfig(session_config=config)
      
      gene_classifier = learn.Estimator(
            model_fn=cnn_model_fn, 
            model_dir=FLAGS.model_dir,
            params={'learning_rate':FLAGS.learning_rate},
            config=config)
            
      input_fn = numpy_io.numpy_input_fn(x={'train_data':train_data}, shuffle=False)
      pred_result = gene_classifier.predict(input_fn = input_fn)
    
  shape = train_labels.shape
  print(shape)
  result = []
  for res in pred_result:
    result.append(res)
  print(len(result))
  result = np.array(result)
  # print(' '.join([str(x) for x in train_labels[20, :]]))
  # print(' '.join([str(x) for x in result[0, :]]))
  hf.savemat('../label/result.mat', 
                {'label':train_labels, 'result':result}, 
                format='7.3')
  auc_result = []
  for i in range(shape[1]):
    if max(train_labels[:, i]) < 1: continue
    auc_score = metrics.roc_auc_score(train_labels[:, i], result[:, i])
    auc_result.append(auc_score)
  print(' '.join([str(x) for x in auc_result]))
  
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument("--batch_size", type=int, default=64, help="Size of batch fit to model")
  parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs to fit the model")
  parser.add_argument("--learning_rate", type=float, default=1.0, help="Learning rate")
  parser.add_argument("--train_algm", type=str, default='Adagrad', help="Algorithms used to train")
  parser.add_argument("--work_type", type=str, default="train", help="Type of operation: train, eval or prid")
  parser.add_argument("--startline", type=int, default=0, help="The start position of work in the data file")
  parser.add_argument("--endline", type=int, default=sys.maxint, help="The end position of work in the data file")
  parser.add_argument(
      "--predict_data",
      type=str,
      default="",
      help="Path to the prediction data.")
  parser.add_argument(
      "--model_dir",
      type=str,
      default="/tmp/gene_deep_finding",
      help="Path to the dirctory store model data")
  parser.add_argument(
      "--log_dir",
      type=str,
      default="/tmp/gene_finding_dir",
      help="Path to the directory store log data")
  parser.add_argument(
      "--log_level",
      type=str,
      default="INFO",
      help="Level to logging")
  parser.add_argument(
      "--ps_hosts",
      type=str,
      default="",
      help="Comma-separated list of hostname:port pairs"
  )
  parser.add_argument(
      "--worker_hosts",
      type=str,
      default="",
      help="Comma-separated list of hostname:port pairs"
  )
  parser.add_argument(
      "--job_name",
      type=str,
      default="",
      help="One of 'ps', 'worker'"
  )
  # Flags for defining the tf.train.Server
  parser.add_argument(
      "--task_index",
      type=int,
      default=0,
      help="Index of task within the job"
  )
  FLAGS, unparsed = parser.parse_known_args()
  
  log_level = FLAGS.log_level
  if log_level == 'INFO' or log_level == 'info':
    tf.logging.set_verbosity(tf.logging.INFO)
  elif log_level == 'ERROR' or log_level == 'error':
    tf.logging.set_verbosity(tf.logging.ERROR)
  work_fun = main
  if FLAGS.work_type == 'train':
    work_fun = main
  elif FLAGS.work_type == 'evaluate':
    work_fun = eval
  elif FLAGS.work_type == 'dtrain':
    work_fun = dtrain
  elif FLAGS.work_type == 'predict':
    work_fun = pred
  tf.app.run(main=work_fun, argv=[sys.argv[0]] + unparsed)