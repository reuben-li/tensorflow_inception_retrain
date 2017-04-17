#!/bin/bash

DATA_PATH="../train_data"
MODEL_PATH="../model"
BOTTLENECK_PATH="../bottleneck"
GRAPH_PATH="../graph"
LABELS_PATH="../labels"

# Compile retrain function in tensorflow master
git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow
./configure
bazel build tensorflow/examples/image_retraining:retrain

# Retraining with proper image directory
bazel-bin/tensorflow/examples/image_retraining/retrain \
  --image_dir "$DATA_PATH" --model_dir "$MODEL_PATH" \
  --bottleneck_dir "$BOTTLENECK_PATH" --output_graph "$GRAPH_PATH" \
  --output_labels "$LABELS_PATH" --validation_batch_size -1 \
  --print_misclassified_test_images True  --test_batch_size -1 \
  --learning-rate 0.02
