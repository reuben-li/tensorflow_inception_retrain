import numpy as np
import tensorflow as tf
import os

testImagePath = './test_data'
modelFullPath = '/tmp/output_graph.pb'
labelsFullPath = '/tmp/output_labels.txt'

def create_graph():
    with tf.gfile.FastGFile(modelFullPath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

def run_inference_on_image():
    answer = None

    create_graph()

    with tf.Session() as sess:
        for subd in os.listdir(testImagePath):
            subPath = os.path.join(testImagePath, subd)
            for image in os.listdir(subPath):
                oneImage = os.path.join(subPath, image)
                image_data = tf.gfile.FastGFile(oneImage, 'rb').read()

                softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
                predictions = sess.run(softmax_tensor,
                                       {'DecodeJpeg/contents:0': image_data})
                predictions = np.squeeze(predictions)

                top_k = predictions.argsort()[-1:][::-1]  # Getting top 5 predictions
                f = open(labelsFullPath, 'rb')
                lines = f.readlines()
                labels = [str(w).replace("\n", "") for w in lines]
                for node_id in top_k:
                    human_string = labels[node_id]
                    score = predictions[node_id]
                    # print('%s (score = %.5f)' % (human_string, score))

                answer = labels[top_k[0]]
                print answer
        return answer

if __name__ == '__main__':
    print run_inference_on_image()
