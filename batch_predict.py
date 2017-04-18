"""
Batch prediction using trained graph
"""
from __future__ import print_function
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

IMAGE_PATH = './test_data'
MODEL_FILE = './output/output_graph.pb'
LABELS_FILE = './output/output_labels.txt'
CLASSES = 24
TEST_FLAG = False

def load_graph():
    """ Load graph into memory"""
    with tf.gfile.FastGFile(MODEL_FILE, 'rb') as graph:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(graph.read())
        _ = tf.import_graph_def(graph_def, name='')

def batch_predict():
    """ Loop through images and run prediction """
    load_graph()
    result = [x[:] for x in [[0] * CLASSES] * CLASSES]
    with tf.Session() as sess:
        for true_label in os.listdir(IMAGE_PATH):
            sub_path = os.path.join(IMAGE_PATH, true_label)
            count = 0
            class_count = 0
            print('Training label ' + str(true_label) + '.')
            for image in os.listdir(sub_path):
                one_image = os.path.join(sub_path, image)
                image_data = tf.gfile.FastGFile(one_image, 'rb').read()
                softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
                predictions = sess.run(softmax_tensor,
                                       {'DecodeJpeg/contents:0': image_data})
                predictions = np.squeeze(predictions)
                top_k = predictions.argsort()[-1:][::-1]
                labels_file = open(LABELS_FILE, 'rb')
                lines = labels_file.readlines()
                labels = [str(w).replace('\n', '') for w in lines]
                answer = int(labels[top_k[0]])
                result[int(true_label)][answer] += 1
                count += 1
                if count >= 2 and TEST_FLAG:
                    break
            print(str(class_count) + '/' + str(CLASSES) + ' trained.')
    return result

def main():
    """ main function for scoping """
    output = batch_predict()
    pos = 0.0
    cnt = 0.0
    sum_avg = 0.0
    for label in xrange(0, CLASSES):
        local_pos = float(output[label][label])
        local_cnt = float(sum(output[label]))
        pos += local_pos
        cnt += local_cnt
        avg = local_pos/local_cnt
        sum_avg += avg
        print('label ' + str(label) + ': ' + str(avg))
    print('acc: ' + str(pos/cnt))
    print('ba: ' + str(sum_avg / CLASSES))
    fig = plt.figure(figsize=(6, 3.2))
    axes = fig.add_subplot(111)
    axes.set_title('colorMap')
    plt.imshow(output, cmap='Blues')
    axes.set_aspect('equal')
    plt.colorbar(orientation='vertical')
    plt.show()
    np.save('output.npy', output)


if __name__ == '__main__':
    main()
