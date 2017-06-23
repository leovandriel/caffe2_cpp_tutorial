#!/bin/bash

set -e

mkdir -p res
mkdir -p tmp

test -f res/image_file.jpg || echo "downloading test image (1)"
test -f res/image_file.jpg || curl --progress-bar --output res/image_file.jpg https://cdn.pixabay.com/photo/2015/02/10/21/28/flower-631765_1280.jpg

test -f res/alexnet_init_net.pb || echo "downloading AlexNet model (2)"
test -f res/alexnet_predict_net.pb || curl --progress-bar --output res/alexnet_predict_net.pb https://s3.amazonaws.com/caffe2/models/bvlc_alexnet/predict_net.pb
test -f res/alexnet_init_net.pb || curl --progress-bar --output res/alexnet_init_net.pb https://s3.amazonaws.com/caffe2/models/bvlc_alexnet/init_net.pb

test -f res/squeeze_init_net.pb || echo "downloading Squeezenet model (2)"
test -f res/squeeze_predict_net.pb || curl --progress-bar --output res/squeeze_predict_net.pb https://s3.amazonaws.com/caffe2/models/squeezenet/predict_net.pb
test -f res/squeeze_init_net.pb || curl --progress-bar --output res/squeeze_init_net.pb https://s3.amazonaws.com/caffe2/models/squeezenet/init_net.pb

test -f res/googlenet_init_net.pb || echo "downloading GoogleNet model (2)"
test -f res/googlenet_predict_net.pb || curl --progress-bar --output res/googlenet_predict_net.pb https://s3.amazonaws.com/caffe2/models/bvlc_googlenet/predict_net.pb
test -f res/googlenet_init_net.pb || curl --progress-bar --output res/googlenet_init_net.pb https://s3.amazonaws.com/caffe2/models/bvlc_googlenet/init_net.pb

test -d res/mnist-train-nchw-leveldb || echo "downloading MNIST train res (2)"
test -f res/train-images-idx3-ubyte || curl --progress-bar http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz | gunzip > res/train-images-idx3-ubyte
test -f res/train-labels-idx1-ubyte || curl --progress-bar http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz | gunzip > res/train-labels-idx1-ubyte
test -d res/mnist-train-nchw-leveldb || make_mnist_db --image_file=res/train-images-idx3-ubyte --label_file=res/train-labels-idx1-ubyte --output_file=res/mnist-train-nchw-leveldb --channel_first --db leveldb

test -d res/mnist-test-nchw-leveldb || echo "downloading MNIST test res (2)"
test -f res/t10k-images-idx3-ubyte || curl --progress-bar http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz | gunzip > res/t10k-images-idx3-ubyte
test -f res/t10k-labels-idx1-ubyte || curl --progress-bar http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz | gunzip > res/t10k-labels-idx1-ubyte
test -d res/mnist-test-nchw-leveldb || make_mnist_db --image_file=res/t10k-images-idx3-ubyte --label_file=res/t10k-labels-idx1-ubyte --output_file=res/mnist-test-nchw-leveldb --channel_first --db leveldb
