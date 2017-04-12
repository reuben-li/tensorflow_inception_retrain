import numpy as np
import tensorflow as tf
import glob
import os
graph_path = '/tmp/output_graph.pb'
image_dir = 'test_data/*/*.jpg'

def load_graph():
    with tf.gfile.FastGFile(graph_path, 'rb') as graph:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(graph.read())
        _ = tf.import_graph_def(graph_def, name='')

testimages=glob.glob(image_dir)
all_predictions = np.zeros(shape=(len(testimages), 25))
load_graph()

with tf.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    for i in range(len(testimages)):
        image_data = tf.gfile.FastGFile(testimages[i], 'rb').read()
        predictions = sess.run(softmax_tensor,
                {'DecodeJpeg/contents:0': image_data})
        all_predictions[i,:] = np.squeeze(predictions)
        if i % 100 == 0:
            print(str(i) +' of a total of '+ str(len(testimages)))
