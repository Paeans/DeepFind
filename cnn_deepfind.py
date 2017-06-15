from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

import numpy as np
import tensorflow as tf

from tensorflow.contrib import learn
from tensorflow.contrib import layers
from tensorflow.contrib.learn import io as numpy_io
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

from gene_encoder import encode_gene
from label_encoder import encode_label

FLAGS = None

kernel_num = [320, 480, 960]
out_unit = 918


model_regular = layers.sum_regularizer(
        [tf.contrib.layers.l1_regularizer(1e-08),
         tf.contrib.layers.l2_regularizer(5e-07)])


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
      kernel_size=[4, 8],
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
      weights_regularizer=model_regular)

  full_connect2 = layers.fully_connected(
      inputs=full_connect1,
      num_outputs=out_unit,
      activation_fn=None,
      weights_regularizer=model_regular)
      
  logits = tf.sigmoid(full_connect2, name='logits_results')
  
  loss = None
  train_op = None

  # Calculate Loss (for both TRAIN and EVAL modes)
  if mode != learn.ModeKeys.INFER:
    #onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    print(labels, logits)
    loss = tf.losses.sigmoid_cross_entropy(
        multi_class_labels=labels, logits=logits)
  
  # Configure the Training Op (for TRAIN mode)
  if mode == learn.ModeKeys.TRAIN:
    train_op = layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        learning_rate=params['learning_rate'], #0.001,
        optimizer="SGD")

  # Generate Predictions
  
  predictions = {
      "logitresult": logits
  }#Generate predictions_key_list, 1:918

  # Return a ModelFnOps object
  return model_fn_lib.ModelFnOps(
      mode=mode, predictions=predictions, loss=loss, train_op=train_op)


def load_train_data():
  #gene_train_feature = json.load(open(FLAGS.train_data, 'r'))
  #gene_train_label = json.load(open(FLAGS.train_label, 'r'))
  
  gene_train_feature = encode_gene(FLAGS.train_data)
  gene_train_label = encode_label(FLAGS.train_label)
  train_data, train_label = [], []
  for key in gene_train_feature.keys():
    if not key in gene_train_label: continue
    if np.random.rand() < 0.8: continue
    train_data.append(gene_train_feature[key])
    train_label.append(gene_train_label[key])
  return np.array(train_data, dtype=np.float32), \
            np.array(train_label, dtype=np.float32)
  #return tf.constant(np.array(data, dtype=np.float32)), \
  #         tf.constant(np.array(label, dtype=np.float32))
  
def load_eval_data():
  gene_eval_feature = encode_gene(FLAGS.eval_data)
  gene_eval_label = encode_label(FLAGS.eval_label)
  
  eval_data, eval_label = [], []
  for key in gene_eval_feature.keys():
    if not key in gene_eval_label: continue    
    eval_data.append(gene_eval_feature[key])
    eval_label.append(gene_eval_label[key])
  return np.array(eval_data, dtype=np.float32), \
           np.array(eval_label, dtype=np.float32)
  #return tf.constant(np.array(data, dtype=np.float32)), \
  #         tf.constant(np.array(label, dtype=np.float32))

def main(unused_argv):
  gene_classifier = learn.Estimator(
          model_fn=cnn_model_fn, 
          model_dir=FLAGS.model_dir,
          params={'learning_rate':0.001})
  
  # Set up logging for predictions
  # Log the values in the "logits" tensor with label "logits_results"
  tensors_to_log = {"logitresult": "logits_results"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)
  # Train the model
  
  train_data, train_labels = load_train_data()
  print(train_data.shape, train_labels.shape)
  
  
  input_fn = numpy_io.numpy_input_fn(
      {'train_data':train_data}, 
      train_labels,
      batch_size=FLAGS.batch_size,
      shuffle=True,
      num_epochs=FLAGS.num_epochs)
  '''
  gene_classifier.fit(
      #input_fn=load_train_data,
      x=train_data,
      y=train_labels,
      batch_size=FLAGS.batch_size,
      steps=train_data.shape[0] / FLAGS.batch_size * FLAGS.num_epochs,  #sizeofdata/batchsize * epoch
      monitors=[logging_hook])
  '''
  gene_classifier.fit(
      input_fn=input_fn,
      #steps=FLAGS.steps_size,  #sizeofdata/batchsize * epoch
      monitors=[logging_hook])
  
  print("Finish training")

def my_auc(labels, predictions, weights=None, num_thresholds=200,
        metrics_collections=None, updates_collections=None,
        curve='ROC', name=None):
  print('shape of label: ', labels.get_shape().ndims, 'shape of predictions: ', predictions.get_shape().ndims)
  #return tf.metrics.auc(labels, predictions, weights, num_thresholds, metrics_collections,
  #      updates_collections, curve, name)
  return tf.metrics.auc(labels[:, 1], predictions[:, 1], weights, num_thresholds, metrics_collections,
        updates_collections, curve, name)
        
def eval(unused_argv):
  gene_classifier = learn.Estimator(
          model_fn=cnn_model_fn,
          model_dir=FLAGS.model_dir)
  
  # Configure the accuracy metric for evaluation
  metrics = {
      "auc":
          learn.MetricSpec(
              metric_fn=my_auc, prediction_key="logitresult"),
  }   #checkpoint_path

  # Evaluate the model and print results
  eval_data, eval_labels = load_eval_data()
  print(eval_data.shape, eval_labels.shape)
  eval_results = gene_classifier.evaluate(
      x=eval_data, y=eval_labels, metrics=metrics)#batch_size=10000, steps=1, 
  print('evaluate results: ', eval_results)
  
  #prid_results = gene_classifier.predict(
  #    x=eval_data, batch_size = 10)
  #print(prid_results.next()['logitresult'])
  
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument("--batch_size", type=int, default=100, help="Size of batch fit to model")
  parser.add_argument("--steps_size", type=int, default=100000, help="Size of batch fit to model")
  parser.add_argument("--num_epochs", type=int, default=2, help="Number of epochs to fit the model")
  parser.add_argument("--work_type", type=str, default="train", help="Type of operation: train, eval or prid")
  parser.add_argument(
      "--train_data", type=str, default="chr21", help="Path to the training data.")
  parser.add_argument(
      "--train_label", type=str, default="chr21", help="Path to the training data.")
  parser.add_argument(
      "--eval_data", type=str, default="chr22", help="Path to the training data.")
  parser.add_argument(
      "--eval_label", type=str, default="chr22", help="Path to the training data.")
  parser.add_argument(
      "--test_data", type=str, default="", help="Path to the test data.")
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
  FLAGS, unparsed = parser.parse_known_args()
  
  log_level = FLAGS.log_level
  if log_level == 'INFO' or log_level == 'info':
    tf.logging.set_verbosity(tf.logging.INFO)
  elif log_level == 'ERROR' or log_level == 'error':
    tf.logging.set_verbosity(tf.logging.ERROR)
  
  if FLAGS.work_type == 'train':
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
  elif FLAGS.work_type == 'eval':
    tf.app.run(main=eval, argv=[sys.argv[0]] + unparsed)
  
'''
total_loss = meansq #or other loss calcuation
l1_regularizer = tf.contrib.layers.l1_regularizer(
   scale=0.005, scope=None
)
weights = tf.trainable_variables() # all vars of your graph
regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, weights)

regularized_loss = total_loss + regularization_penalty # this loss needs to be minimized
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(regularized_loss)
'''

'''
input_fn = tf.contrib.learn.io.numpy_input_fn({"x":x}, y, batch_size=4,
                                              num_epochs=1000)
'''

'''
regressor.fit(input_fn=lambda: input_fn(training_set), steps=5000)
'''


