import tensorflow as  tf

def weight_var(shape):
    with tf.name_scope("weights"):
        initial = tf.truncated_normal(shape, stddev=0.1)
        initial = tf.Variable(initial)
        tf.summary.histogram('weights', initial)
        return initial


def bias_var(shape, name = None):
    with tf.name_scope("biases"):
        initial = tf.constant(0.1, shape=shape)
        if name:
            initial = tf.Variable(initial, name=name)
        else:
            initial = tf.Variable(initial)
        tf.summary.histogram('biases', initial)
        return initial


def conv2d(x, W):
    # 第3，4个参数是x跨度1 y跨度1
    # SAME padding 是补0之后卷积，前后层长宽相同
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")


def max_pool_2x2(x):
    # 使用池化减轻跨度大问题
    # ksize 卷积核大小 2x2
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def inference(x_image):
    with tf.name_scope('hidden_layer_1'):
        W_conv1 = weight_var([5, 5, 1, 32])  # 卷积面patch 5x5 insize=1色深 outsize=32提取32种特征
        b_conv1 = bias_var([32])  # 过滤层层数
        hidden_conv1 = conv2d(x_image, W_conv1) + b_conv1  # outsize 28x28x32
        hidden_conv1 = tf.nn.relu(hidden_conv1)
        hidden_pool1 = max_pool_2x2(hidden_conv1)  # outsize 14x14x32

    with tf.name_scope('hidden_layer_2'):
        W_conv2 = weight_var([5, 5, 32, 64])  # 卷积面patch 5x5 insize=1色深 outsize=32提取32种特征
        b_conv2 = bias_var([64])  # 过滤层层数
        hidden_conv2 = conv2d(hidden_pool1, W_conv2) + b_conv2  # outsize 14x14x64
        hidden_conv2 = tf.nn.relu(hidden_conv2)
        hidden_pool2 = max_pool_2x2(hidden_conv2)  # outsize 7x7x64

    with tf.name_scope('full_connection_1'):
        # full connect
        W_fc1 = weight_var([7 * 7 * 64, 1024])
        b_fc1 = bias_var([1024])
        hidden_pool2_flat = tf.reshape(hidden_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(hidden_pool2_flat, W_fc1))
        h_fc1_drop = tf.nn.dropout(h_fc1, 0.5) # dropout fix 0.5

    with tf.name_scope('full_connection_2'):
        # full connect
        W_fc2 = weight_var([1024, 10])
        b_fc2 = bias_var([10])

    prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    return prediction, b_fc2