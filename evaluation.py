import tensorflow as tf
def accuracy(logits, mnist):
    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(
            tf.argmax(logits, -1), tf.argmax(mnist.test.labels, -1)
        )
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy", accuracy)