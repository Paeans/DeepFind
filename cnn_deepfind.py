from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile
import json

import numpy as np
import tensorflow as tf

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

FLAGS = None

tf.logging.set_verbosity(tf.logging.ERROR)


def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # MNIST images are 28x28 pixels, and have one color channel
  input_layer = tf.reshape(features, [-1, 4, 1000, 1])

  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 4, 1000, 1]
  # Output Tensor Shape: [batch_size, 4, 1000, 32]
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[4, 8],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 4, 1000, 32]
  # Output Tensor Shape: [batch_size, 1, 250, 32]
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[4, 4], strides=[4, 4])

  # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 1, 250, 32]
  # Output Tensor Shape: [batch_size, 1, 250, 64]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[1, 8],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 1, 250, 64]
  # Output Tensor Shape: [batch_size, 1, 125, 64]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[1, 2], strides=[1, 2])

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 1, 125, 64]
  # Output Tensor Shape: [batch_size, 1 * 125 * 64]
  pool2_flat = tf.reshape(pool2, [-1, 1 * 125 * 64])

  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 7 * 7 * 64]
  # Output Tensor Shape: [batch_size, 1024]
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == learn.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 10]
  logits = tf.layers.dense(inputs=dropout, units=918)

  loss = None
  train_op = None

  # Calculate Loss (for both TRAIN and EVAL modes)
  if mode != learn.ModeKeys.INFER:
    #onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == learn.ModeKeys.TRAIN:
    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        learning_rate=0.1,
        optimizer="SGD")

  # Generate Predictions
  print(logits.shape)
  predictions = {
      "classes": tf.argmax(
          input=logits, axis=1),
      "probabilities": tf.nn.softmax(
          logits, name="softmax_tensor"),
      "logitresult": logits
  }

  # Return a ModelFnOps object
  return model_fn_lib.ModelFnOps(
      mode=mode, predictions=predictions, loss=loss, train_op=train_op)


def load_train_data():
  gene_train_feature = json.load(open(FLAGS.train_data, 'r'))
  gene_train_label = json.load(open(FLAGS.train_label, 'r'))
  data, label = [], []
  for key in gene_train_feature.keys():
    if not key in gene_train_label: continue
    data.append(gene_train_feature[key])
    label.append(gene_train_label[key])
  print(len(data), len(label))  
  return np.array(data, dtype=np.float32), np.array(label, dtype=np.float32)
  #return tf.constant(np.array(data, dtype=np.float32)), \
  #         tf.constant(np.array(label, dtype=np.float32))
  
def load_eval_data():
  gene_train_feature = json.load(open(FLAGS.train_data, 'r'))
  gene_train_label = json.load(open(FLAGS.train_label, 'r'))
  data, label = [], []
  for key in gene_train_feature.keys():
    if not key in gene_train_label: continue
    data.append(gene_train_feature[key])
    label.append(gene_train_label[key])
  print(len(data), len(label))
  return np.array(data, dtype=np.float32), np.array([tf.argmax(input=x, axis=1) for x in np.array(label, dtype=np.float32)])
  #return tf.constant(np.array(data, dtype=np.float32)), \
  #         tf.constant(np.array(label, dtype=np.float32))

def main(unused_argv):
  gene_classifier = learn.Estimator(
          model_fn=cnn_model_fn, 
          model_dir=FLAGS.model_dir)
          
  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)
    # Train the model
  '''
  mnist_classifier.fit(
      x=train_data,
      y=train_labels,
      batch_size=100,
      steps=200,
      monitors=[logging_hook])
  '''
  train_data, train_labels = load_train_data()
  print(train_data.shape, train_labels.shape)
  '''
  gene_classifier.fit(
      #input_fn=load_train_data,
      x=train_data,
      y=train_labels,
      batch_size=100,
      steps=2000,  #sizeofdata/batchsize * epoch
      monitors=[logging_hook])
  '''
  # Configure the accuracy metric for evaluation
  metrics = {
      "accuracy":
          learn.MetricSpec(
              metric_fn=tf.metrics.accuracy, prediction_key="classes"),
  }

  # Evaluate the model and print results
  eval_data, eval_labels = load_eval_data()
  #eval_results = gene_classifier.evaluate(
  #    x=eval_data, y=eval_labels, steps=1, metrics=metrics)
  #print(eval_results)
  
  prid_results = gene_classifier.predict(
      x=eval_data, batch_size = 10)
  for r in prid_results:
    print(r['classes'])
  
  
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument(
      "--train_data", type=str, default="./label/chrY-encode.json", help="Path to the training data.")
  parser.add_argument(
    "--train_label", type=str, default="./label/chrY-seg-label.json", help="Path to the training data.")
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
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)