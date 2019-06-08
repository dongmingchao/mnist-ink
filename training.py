import tensorflow as tf

def train(cross_entropy):
    with tf.name_scope("train"):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    return train_step