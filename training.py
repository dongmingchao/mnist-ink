import tensorflow as tf
from inference import *
from input_data import data_type
from config import config

def train_setup(train_labels):
    IMAGE_SIZE = config.IMAGE_SIZE
    NUM_CHANNELS = config.NUM_CHANNELS
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
    return train_size, train_data_node, train_labels_node, eval_data

def train(train_size, train_data_node, loss, logits, eval_data):
    BATCH_SIZE = config.BATCH_SIZE
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
    eval_prediction = tf.nn.softmax(inference(eval_data)[0])
    return learning_rate, optimizer, train_prediction, eval_prediction