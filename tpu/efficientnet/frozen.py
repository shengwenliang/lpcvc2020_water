import model_builder_factory
import tensorflow as tf
import keras

model_builder = model_builder_factory.get_model_builder("efficientnet-lite0")

def restore_model(sess, ckpt_dir, enable_ema=True):
    sess.run(tf.global_variables_initializer())
    checkpoint = tf.train.latest_checkpoint(ckpt_dir)
    checkpoint = "./checkpoint_lite0/model.ckpt-2893563"
    print(checkpoint)
    if enable_ema:
        ema = tf.train.ExponentialMovingAverage(decay=0.0)
        ema_vars = tf.trainable_variables() + tf.get_collection("moving_vars")
        for v in tf.global_variables():
            if "moving_mean" in v.name or "moving_variance" in v.name:
                ema_vars.append(v)
        ema_vars = list(set(ema_vars))
        var_dict = ema.variables_to_restore(ema_vars)
    else:
        var_dict = None

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(var_dict, max_to_keep=1)
    saver.restore(sess, checkpoint)
    tf_session = keras.backend.get_session()
    input_graph_def = tf_session.graph.as_graph_def()
    save_path = saver.save(tf_session, './checkpoint.ckpt')
    tf.train.write_graph(input_graph_def, './', 'efficientnet_lite0.pb', as_text=False)

image_size = 224
with tf.Graph().as_default(), tf.Session() as sess:
    images = tf.placeholder(tf.float32, shape=(None, image_size, image_size, 3), name="images")
    logits, endpoints = model_builder.build_model(images, "efficientnet-lite0",False)
    output_tensor = tf.nn.softmax(logits)
    restore_model(sess, "./dpu_lite0/archive/", True)


meta_path = './checkpoint.ckpt.meta' # Your .meta file
output_node_names = ['logits']    # Output nodes

with tf.Session() as sess:
    # Restore the graph
    saver = tf.train.import_meta_graph(meta_path)

    # Load weights
    saver.restore(sess,tf.train.latest_checkpoint('./'))
    #output_node_names = [n.name for n in tf.get_default_graph().as_graph_def().node]
    #print(output_node_names)
    # Freeze the graph
    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        sess.graph_def,
        output_node_names)

    # Save the frozen graph
    tf.train.write_graph(frozen_graph_def, './', 'frozen.pb', as_text=False)
