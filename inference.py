import tensorflow as tf

# The variables below hold all the trainable weights. They are passed an
# initial value which will be assigned when we call:
# {tf.global_variables_initializer().run()}
conv1_weights = tf.Variable(
    tf.truncated_normal(
        [5, 5, NUM_CHANNELS, 32],  # 5x5 filter, depth 32.
        stddev=0.1,
        seed=SEED,
        dtype=data_type(),
    )
)
conv1_biases = tf.Variable(tf.zeros([32], dtype=data_type()))
conv2_weights = tf.Variable(
    tf.truncated_normal([5, 5, 32, 64], stddev=0.1, seed=SEED, dtype=data_type())
)
conv2_biases = tf.Variable(tf.constant(0.1, shape=[64], dtype=data_type()))
fc1_weights = tf.Variable(  # fully connected, depth 512.
    tf.truncated_normal(
        [IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64, 512],
        stddev=0.1,
        seed=SEED,
        dtype=data_type(),
    )
)
fc1_biases = tf.Variable(tf.constant(0.1, shape=[512], dtype=data_type()))
fc2_weights = tf.Variable(
    tf.truncated_normal([512, NUM_LABELS], stddev=0.1, seed=SEED, dtype=data_type())
)
fc2_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS], dtype=data_type()))