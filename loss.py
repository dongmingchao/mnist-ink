import tensorflow as tf

def loss(prediction, ys):
    cross_entropy = tf.reduce_mean(
        -tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1])
    )  # classification loss
    tf.summary.scalar("loss", cross_entropy)
    return cross_entropy
