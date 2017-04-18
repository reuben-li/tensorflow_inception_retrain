#!/bin/bash

DATA_PATH="../train_data"
MODEL_PATH="../model"
BOTTLENECK_PATH="../bottleneck"
GRAPH_PATH="../output/output_graph.pb"
LABELS_PATH="../output/output_labels.txt"
INIT=false

if [  "$INIT" = true ]; then
  # install bazel (ubuntu)
  echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
  curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -
  sudo apt-get update && sudo apt-get install bazel

  # Compile retrain function in tensorflow master
  git clone https://github.com/tensorflow/tensorflow.git
  cd tensorflow
  ./configure
  bazel build tensorflow/examples/image_retraining:retrain
else
  cd tensorflow
fi

# Retraining with proper image directory
bazel-bin/tensorflow/examples/image_retraining/retrain \
  --image_dir "$DATA_PATH" --model_dir "$MODEL_PATH" \
  --bottleneck_dir "$BOTTLENECK_PATH" --output_graph "$GRAPH_PATH" \
  --output_labels "$LABELS_PATH" --validation_batch_size -1 \
  --print_misclassified_test_images True  --test_batch_size -1 \
  --learning-rate 0.02
