# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Simple, end-to-end, LeNet-5-like convolutional MNIST model example.

This should achieve a test error of 0.7%. Please keep this model as simple and
linear as possible, it is meant as a tutorial for simple convolutional models.
Run with --self_test on the command line to execute a short self-test.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import gzip
import numpy as np
import os
import shutil
import sys
import time

import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from .config import *


def error_rate(predictions, labels):
    """Return the error rate based on dense predictions and sparse labels."""
    return 100.0 - (
        100.0 * numpy.sum(numpy.argmax(predictions, 1) == labels) / predictions.shape[0]
    )


def main(_):
    global BATCH_SIZE
    if FLAGS.self_test:
        print("Running self-test.")
        train_data, train_labels = fake_data(256)
        validation_data, validation_labels = fake_data(EVAL_BATCH_SIZE)
        test_data, test_labels = fake_data(EVAL_BATCH_SIZE)
        num_epochs = 1
    else:
        # Get the data.
        train_data_filename = maybe_download("train-images-idx3-ubyte.gz")
        train_labels_filename = maybe_download("train-labels-idx1-ubyte.gz")
        test_data_filename = maybe_download("t10k-images-idx3-ubyte.gz")
        test_labels_filename = maybe_download("t10k-labels-idx1-ubyte.gz")

        # Extract it into numpy arrays.
        train_data = extract_data(train_data_filename, 60000)
        train_labels = extract_labels(train_labels_filename, 60000)
        test_data = extract_data(test_data_filename, 10000)
        test_labels = extract_labels(test_labels_filename, 10000)

        # Generate a validation set.
        validation_data = train_data[:VALIDATION_SIZE, ...]
        validation_labels = train_labels[:VALIDATION_SIZE]
        train_data = train_data[VALIDATION_SIZE:, ...]
        train_labels = train_labels[VALIDATION_SIZE:]
        num_epochs = NUM_EPOCHS
    train_size = train_labels.shape[0]

    # This is where training samples and labels are fed to the graph.
    # These placeholder nodes will be fed a batch of training data at each
    # training step using the {feed_dict} argument to the Run() call below.
    train_data_node = tf.placeholder(
        data_type(),
        shape=(None, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS),
        name="image_input",
    )
    train_labels_node = tf.placeholder(tf.int64, shape=(None,))
    eval_data = tf.placeholder(
        data_type(), shape=(None, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
    )

    # We will replicate the model structure for the training subgraph, as well
    # as the evaluation subgraphs, while sharing the trainable parameters.
    def model(data, train=False):
        """The Model definition."""
        # 2D convolution, with 'SAME' padding (i.e. the output feature map has
        # the same size as the input). Note that {strides} is a 4D array whose
        # shape matches the data layout: [image index, y, x, depth].
        conv = tf.nn.conv2d(data, conv1_weights, strides=[1, 1, 1, 1], padding="SAME")
        # Bias and rectified linear non-linearity.
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
        # Max pooling. The kernel size spec {ksize} also follows the layout of
        # the data. Here we have a pooling window of 2, and a stride of 2.
        pool = tf.nn.max_pool(
            relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME"
        )
        conv = tf.nn.conv2d(pool, conv2_weights, strides=[1, 1, 1, 1], padding="SAME")
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
        pool = tf.nn.max_pool(
            relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME"
        )
        # Reshape the feature map cuboid into a 2D matrix to feed it to the
        # fully connected layers.
        pool_shape = pool.get_shape().as_list()
        reshape = tf.reshape(
            pool, [tf.shape(pool)[0], pool_shape[1] * pool_shape[2] * pool_shape[3]]
        )
        # Fully connected layer. Note that the '+' operation automatically
        # broadcasts the biases.
        hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
        # Add a 50% dropout during training only. Dropout also scales
        # activations such that no rescaling is needed at evaluation time.
        if train:
            hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
        return tf.matmul(hidden, fc2_weights) + fc2_biases

    # Training computation: logits + cross-entropy loss.
    logits = model(train_data_node, True)
    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=train_labels_node, logits=logits
        )
    )
    predict_op = tf.argmax(logits, 1, name="predict_op")

    # L2 regularization for the fully connected parameters.
    regularizers = (
        tf.nn.l2_loss(fc1_weights)
        + tf.nn.l2_loss(fc1_biases)
        + tf.nn.l2_loss(fc2_weights)
        + tf.nn.l2_loss(fc2_biases)
    )
    # Add the regularization term to the loss.
    loss += 5e-4 * regularizers

    # Optimizer: set up a variable that's incremented once per batch and
    # controls the learning rate decay.
    batch = tf.Variable(0, dtype=data_type())
    # Decay once per epoch, using an exponential schedule starting at 0.01.
    learning_rate = tf.train.exponential_decay(
        0.01,  # Base learning rate.
        batch * BATCH_SIZE,  # Current index into the dataset.
        train_size,  # Decay step.
        0.95,  # Decay rate.
        staircase=True,
    )
    # Use simple momentum for the optimization.
    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(
        loss, global_step=batch
    )

    # Predictions for the current training minibatch.
    train_prediction = tf.nn.softmax(logits)

    # Predictions for the test and validation, which we'll compute less often.
    eval_prediction = tf.nn.softmax(model(eval_data))

    # Small utility function to evaluate a dataset by feeding batches of data to
    # {eval_data} and pulling the results from {eval_predictions}.
    # Saves memory and enables this to run on smaller GPUs.
    def eval_in_batches(data, sess):
        """Get all predictions for a dataset by running it in small batches."""
        size = data.shape[0]
        if size < EVAL_BATCH_SIZE:
            raise ValueError("batch size for evals larger than dataset: %d" % size)
        predictions = numpy.ndarray(shape=(size, NUM_LABELS), dtype=numpy.float32)
        for begin in xrange(0, size, EVAL_BATCH_SIZE):
            end = begin + EVAL_BATCH_SIZE
            if end <= size:
                predictions[begin:end, :] = sess.run(
                    eval_prediction, feed_dict={eval_data: data[begin:end, ...]}
                )
            else:
                batch_predictions = sess.run(
                    eval_prediction, feed_dict={eval_data: data[-EVAL_BATCH_SIZE:, ...]}
                )
                predictions[begin:, :] = batch_predictions[begin - size :, :]
        return predictions

    ### Change original code
    # Add model_dir to save model
    if not os.path.exists(FLAGS.model_dir):
        os.makedirs(FLAGS.model_dir)

    ### Change original code
    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()
    # Create a builder for writing saved model for serving.
    if os.path.isdir(FLAGS.export_dir):
        shutil.rmtree(FLAGS.export_dir)
    builder = tf.saved_model.builder.SavedModelBuilder(FLAGS.export_dir)

    # Create a local session to run the training.
    start_time = time.time()
    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
        # Run all the initializers to prepare the trainable parameters.
        tf.global_variables_initializer().run()
        ### Change original code
        # Save checkpoint when training
        ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print("Load from " + ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)

        ### Change original code
        # Create summary, logs will be saved, which can display in Tensorboard
        tf.summary.scalar("loss", loss)
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(os.path.join(FLAGS.model_dir, "log"), sess.graph)

        print("Initialized!")

        # Loop through training steps.
        for step in xrange(int(num_epochs * train_size) // BATCH_SIZE):
            # Compute the offset of the current minibatch in the data.
            # Note that we could use better randomization across epochs.
            offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
            batch_data = train_data[offset : (offset + BATCH_SIZE), ...]
            batch_labels = train_labels[offset : (offset + BATCH_SIZE)]
            # This dictionary maps the batch data (as a numpy array) to the
            # node in the graph it should be fed to.
            feed_dict = {train_data_node: batch_data, train_labels_node: batch_labels}
            # Run the optimizer to update weights.
            sess.run(optimizer, feed_dict=feed_dict)
            # print some extra information once reach the evaluation frequency
            if step % EVAL_FREQUENCY == 0:
                # fetch some extra nodes' data
                ### Change original code
                # Add summary
                summary, l, lr, predictions = sess.run(
                    [merged, loss, learning_rate, train_prediction], feed_dict=feed_dict
                )
                writer.add_summary(summary, step)
                elapsed_time = time.time() - start_time
                start_time = time.time()
                ### Change original code
                # save model
                if step % (EVAL_FREQUENCY * 10) == 0:
                    saver.save(
                        sess,
                        os.path.join(FLAGS.model_dir, "model.ckpt"),
                        global_step=step,
                    )
                print(
                    "Step %d (epoch %.2f), %.1f ms"
                    % (
                        step,
                        float(step) * BATCH_SIZE / train_size,
                        1000 * elapsed_time / EVAL_FREQUENCY,
                    )
                )
                print("Minibatch loss: %.3f, learning rate: %.6f" % (l, lr))
                print("Minibatch error: %.1f%%" % error_rate(predictions, batch_labels))
                print(
                    "Validation error: %.1f%%"
                    % error_rate(
                        eval_in_batches(validation_data, sess), validation_labels
                    )
                )
                sys.stdout.flush()

        ### Change original code
        # Save model
        inputs = {tf.saved_model.signature_constants.PREDICT_INPUTS: train_data_node}
        outputs = {tf.saved_model.signature_constants.PREDICT_OUTPUTS: predict_op}
        serving_signatures = {
            "Infer": tf.saved_model.signature_def_utils.predict_signature_def(  # tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                inputs, outputs
            )
        }
        builder.add_meta_graph_and_variables(
            sess,
            [tf.saved_model.tag_constants.SERVING],
            signature_def_map=serving_signatures,
            assets_collection=tf.get_collection(tf.GraphKeys.ASSET_FILEPATHS),
            clear_devices=True,
        )
        builder.save()

        # Finally print the result!
        test_error = error_rate(eval_in_batches(test_data, sess), test_labels)
        print("Test error: %.1f%%" % test_error)
        if FLAGS.self_test:
            print("test_error", test_error)
            assert test_error == 0.0, "expected 0.0 test_error, got %.2f" % (
                test_error,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_fp16",
        default=False,
        help="Use half floats instead of full floats if True.",
        action="store_true",
    )
    parser.add_argument(
        "--self_test",
        default=False,
        action="store_true",
        help="True if running a self test.",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="input",
        help="Directory to put the input data.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="output",
        help="Directory to put the checkpoint files.",
    )
    parser.add_argument(
        "--export_dir",
        type=str,
        default="export",
        help="Directory to put the savedmodel files.",
    )

    FLAGS, unparsed = parser.parse_known_args()
    WORK_DIRECTORY = FLAGS.input_dir
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

