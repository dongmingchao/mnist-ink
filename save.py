import tensorflow as tf
import os
import shutil

def save_model(export_dir, train_data_node, predict_op, sess):
    if os.path.exists(export_dir):
        shutil.rmtree(export_dir)
    ### Change original code
    # Save model
    builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
    inputs = {tf.saved_model.signature_constants.PREDICT_INPUTS: train_data_node}
    outputs = {tf.saved_model.signature_constants.PREDICT_OUTPUTS: predict_op}
    print('input key', tf.saved_model.signature_constants.PREDICT_INPUTS)
    print('output key', tf.saved_model.signature_constants.PREDICT_OUTPUTS)
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