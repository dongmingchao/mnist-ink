import tensorflow as tf
import numpy as np
from tensorflow.python.platform import gfile
from tensorflow.python.saved_model import tag_constants
from tensorflow.examples.tutorials.mnist import input_data

# signature_key = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
signature_key = 'Infer'
export_dir = 'export'
input_key = 'inputs'
output_key = 'outputs'
mnist = input_data.read_data_sets("input", one_hot=True)

with tf.Session() as sess:
        meta_graph_def = tf.saved_model.loader.load(sess, [tag_constants.SERVING], export_dir)
        signature = meta_graph_def.signature_def
        print(signature)
        x_tensor_name = signature[signature_key].inputs[input_key].name
        y_tensor_name = signature[signature_key].outputs[output_key].name

        x = sess.graph.get_tensor_by_name(x_tensor_name)
        y = sess.graph.get_tensor_by_name(y_tensor_name)

        batch_xs, batch_ys = mnist.test.next_batch(1)
        batch_xs = np.reshape(batch_xs, [-1, 28, 28, 1])
        y_out = sess.run(y, feed_dict = {x: batch_xs})
        print('---------- inference results ----------------')
        print(y_out, batch_ys)
        batch_xs, batch_ys = mnist.test.next_batch(1000)
        batch_xs = np.reshape(batch_xs, [-1, 28, 28, 1])
        y_out = sess.run(y, feed_dict = {x: batch_xs})
        batch_ys = np.argmax(batch_ys, 1)
        score = 0
        for each in range(1000):
            if y_out[each] == batch_ys[each]:
                score += 1
        print('Validation error: %.1f%%' % (score/1000*100))