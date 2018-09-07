from __future__ import print_function
import numpy as np
import os
import csv
import tensorflow as tf
from PIL import Image
from scipy import misc
from sklearn.model_selection import train_test_split


EPOCHS = 10
BATCH_SIZE = 16
LABELS = ["Acute_Scar", "Micronuclei", "Subacute_Scar", "Unscarred"]
DATA_INPUT = "../../data_scripts/nuclei.csv"
OUTPUT_DIRECTORY = "../Models/model1"


feature_data, label_data = [],[]
input_file = open(DATA_INPUT, "r")
reader = csv.reader(input_file)
current = os.getcwd()
os.chdir(os.path.dirname(DATA_INPUT))
for row in reader:
	# features.append(row[0])
	feature_data.append(misc.imread(row[0]))
	label_data.append(LABELS.index(row[1]))

os.chdir(current)

feature_data = np.array(feature_data, dtype = np.float32)
label_data = np.array(label_data, dtype = np.int32)
X_train, X_test, y_train, y_test = train_test_split(feature_data,label_data,test_size=0.33, random_state = 42)	

# # step 2: create a dataset returning slices of `filenames`
dataset = tf.data.Dataset.from_tensor_slices((feature_data, label_data)).repeat().batch(BATCH_SIZE)

# step 4: create iterator and final input tensor
iterator = dataset.make_one_shot_iterator()
features, labels = iterator.get_next()



tf.logging.set_verbosity(tf.logging.INFO)

# Our application logic will be added here
def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  input_layer = tf.reshape(features["x"], [-1, 52, 52, 1])
  print(input_layer)

  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2 and Pooling Layer #2
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Dense Layer
  pool2_flat = tf.reshape(pool2, [-1, 13 * 13 * 64])
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits Layer
  logits = tf.layers.dense(inputs=dropout, units=10)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main(unused_argv):
  # Load training and eval data

  # Create the Estimator
  mnist_classifier = tf.estimator.Estimator(
	model_fn=cnn_model_fn, model_dir=OUTPUT_DIRECTORY)
  # Set up logging for predictions
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)
  # Train the model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": X_train},
    y=y_train,
    batch_size=100,
    num_epochs=None,
    shuffle=True)
  mnist_classifier.train(
    input_fn=train_input_fn,
    steps=1000, #was 20000 got 64% accuracy with 1000 steps
    hooks=[logging_hook])
  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
	x={"x": X_test},
	y=y_test,
	num_epochs=1,
	shuffle=False)
  eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)

if __name__ == "__main__":
  tf.app.run()


