```
git clone tensorflow
cd tensorflow
./configure
bazel build tensorflow/examples/image_retraining:retrain
bazel-bin/tensorflow/examples/image_retraining/retrain --image_dir ../train_data_small --model_dir ../model  --bottleneck_dir ../bottleneck
```

```
bazel build tensorflow/examples/label_image:label_image && \
bazel-bin/tensorflow/examples/label_image/label_image \
--graph=/tmp/output_graph.pb --labels=/tmp/output_labels.txt \
--output_layer=final_result \
--image=$HOME/flower_photos/daisy/21652746_cc379e0eea_m.jpg
```

```
bazel-bin/tensorflow/examples/image_retraining/retrain --image_dir ../train_data/ --validation_batch_size -1
```

```
python build_image_data.py --train_directory=./train --output_directory=./  \
--validation_directory=./validate --labels_file=mylabels.txt   \
--train_shards=1 --validation_shards=1 --num_threads=1 
```

```
tensorboard --logdir /tmp/retrain_logs/
```

```
bazel-bin/tensorflow/examples/image_retraining/retrain --image_dir ../train_data/ --validation_batch_size -1 --print_misclassified_test_images True  --test_batch_size -1 --learning-rate 0.02
```

```
bazel-bin/tensorflow/examples/label_image/label_image --graph=/tmp/output_graph.pb --labels=/tmp/output_labels.txt --output_layer=final_result --image=../test_data/5/125F009A_151181_09_A1.jpg --input_layer=Mul
```

```
for i in *; do ls $i | while read j; do ../tensorflow/bazel-bin/tensorflow/examples/label_image/label_image --graph=/tmp/output_graph.pb --labels=/tmp/output_labels.txt --output_layer=final_result --image=$i/$j --input_layer=Mul; done; done
```
