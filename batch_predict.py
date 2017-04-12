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
        global_count = 0.0
        global_pos = 0.0
        acc_list = []

        for subd in os.listdir(testImagePath):
            subPath = os.path.join(testImagePath, subd)
            acc_list.append(0)
            for image in os.listdir(subPath):
                oneImage = os.path.join(subPath, image)
                image_data = tf.gfile.FastGFile(oneImage, 'rb').read()

                softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
                predictions = sess.run(softmax_tensor,
                                       {'DecodeJpeg/contents:0': image_data})
                predictions = np.squeeze(predictions)

                top_k = predictions.argsort()[-1:][::-1]
                f = open(labelsFullPath, 'rb')
                lines = f.readlines()
                labels = [str(w).replace("\n", "") for w in lines]
                for node_id in top_k:
                    human_string = labels[node_id]
                    score = predictions[node_id]
                answer = labels[top_k[0]]
                global_count += 1.0
                if answer == subd:
                    global_pos += 1.0
                acc = round(global_pos/global_count, 3)
                acc_list[int(subd)] = acc
                gc = int(global_count)
                ba = round(sum(acc_list)/float(len(acc_list)), 3)
                print(str(gc) + '. acc: ' + str(acc) + '; bal_acc: ' + str(ba))
    return

if __name__ == '__main__':
    print run_inference_on_image()
