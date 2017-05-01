"""
Batch prediction using trained graph
"""
from __future__ import print_function
import os
import itertools
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

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
    pred_list = []
    label_list = []
    class_names = []
    with tf.Session() as sess:
        class_count = 0
        for true_label in sorted(os.listdir(IMAGE_PATH), key=int):
            sub_path = os.path.join(IMAGE_PATH, true_label)
            count = 0
            class_names.append(true_label)
            print('Predicting label ' + str(true_label) + '.')
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
                label_list.append(int(true_label))
                pred_list.append(int(answer))
                count += 1
                if count >= 2 and TEST_FLAG:
                    break
            class_count += 1
            print(str(class_count) + '/' + str(CLASSES) + ' predicted.')
    return label_list, pred_list, class_names

def acc(dataf):
    """ Find the average score for a given dataframe """
    return float(len(dataf[(dataf['label'] == dataf['pred'])]))/len(dataf)

def balanced_acc(dataf):
    """ Find the balanced average score for a given dataframe """
    return sum(dataf.groupby('label').apply(acc))/CLASSES

def plot_confusion_matrix(matrix, classes, normalize=False,
                          title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = matrix.max() / 2.
    for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
        plt.text(j, i, round(matrix[i, j], 1),
                 horizontalalignment="center",
                 color="white" if matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def main():
    """ main function for scoping """
    label_list, pred_list, class_names = batch_predict()
    dataframe = pd.DataFrame({'label': label_list, 'pred': pred_list})

    print('accuracy: ' + str(acc(dataframe)))
    print('balanced_accuracy: ' + str(balanced_acc(dataframe)))

    output = confusion_matrix(label_list, pred_list)
    np.set_printoptions(precision=2)

    plt.figure(figsize=(18, 10))
    plot_confusion_matrix(output, classes=class_names,
                          title='Confusion matrix, without normalization')

    plt.figure(figsize=(18, 10))
    plot_confusion_matrix(output, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')
    plt.show()
    np.save('output.npy', output)

if __name__ == '__main__':
    main()
