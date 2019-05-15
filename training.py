import tensorflow as tf
from .inference import *

# Training computation: logits + cross-entropy loss.
logits = inference(train_data_node, True)
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
