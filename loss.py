import tensorflow as tf

def get_loss(train_labels_node, logits, layers):
    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=train_labels_node, logits=logits
        )
    )
    predict_op = tf.argmax(logits, 1, name="predict_op")

    # L2 regularization for the fully connected parameters.
    regularizers = 0
    for each in layers:
        regularizers += tf.nn.l2_loss(each)
        # tf.nn.l2_loss(fc1_weights)
        # + tf.nn.l2_loss(fc1_biases)
        # + tf.nn.l2_loss(fc2_weights)
        # + tf.nn.l2_loss(fc2_biases)
    # Add the regularization term to the loss.
    loss += 5e-4 * regularizers
    return loss, predict_op