#!/bin/bash

set -e

mkdir -p res
mkdir -p tmp

test -f res/image_file.jpg || echo "downloading test image (1)"
test -f res/image_file.jpg || curl --progress-bar --output res/image_file.jpg https://cdn.pixabay.com/photo/2015/02/10/21/28/flower-631765_1280.jpg

test -f res/squeezenet_init_net.pb || echo "downloading Squeezenet model (2)"
test -f res/squeezenet_predict_net.pb || curl --progress-bar --location --output res/squeezenet_predict_net.pb https://github.com/caffe2/models/raw/master/squeezenet/predict_net.pb
test -f res/squeezenet_init_net.pb || curl --progress-bar --location --output res/squeezenet_init_net.pb https://github.com/caffe2/models/raw/master/squeezenet/exec_net.pb

test -d res/mnist-train-nchw-leveldb || echo "downloading MNIST train data (2)"
test -f res/train-images-idx3-ubyte || curl --progress-bar http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz | gunzip > res/train-images-idx3-ubyte
test -f res/train-labels-idx1-ubyte || curl --progress-bar http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz | gunzip > res/train-labels-idx1-ubyte
test -d res/mnist-train-nchw-leveldb || make_mnist_db --image_file=res/train-images-idx3-ubyte --label_file=res/train-labels-idx1-ubyte --output_file=res/mnist-train-nchw-leveldb --channel_first --db leveldb

test -d res/mnist-test-nchw-leveldb || echo "downloading MNIST test data (2)"
test -f res/t10k-images-idx3-ubyte || curl --progress-bar http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz | gunzip > res/t10k-images-idx3-ubyte
test -f res/t10k-labels-idx1-ubyte || curl --progress-bar http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz | gunzip > res/t10k-labels-idx1-ubyte
test -d res/mnist-test-nchw-leveldb || make_mnist_db --image_file=res/t10k-images-idx3-ubyte --label_file=res/t10k-labels-idx1-ubyte --output_file=res/mnist-test-nchw-leveldb --channel_first --db leveldb

test -f res/shakespeare.txt || echo "downloading RNN and LSTM test data (1)"
test -f res/shakespeare.txt || curl --progress-bar --output res/shakespeare.txt https://caffe2.ai/static/datasets/shakespeare.txt

test -d res/images || echo "downloading CNN image test data (2)"
test -d res/images || mkdir -p res/download/cats
test -d res/images || cd res/download/cats
test -d res/images || curl  --progress-bar http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n02121808 | tail -n 50 | tr -d '\r' | xargs -L1 -P 5 -I {} sh -c "curl --connect-timeout 10 --fail --silent --progress-bar --remote-name '{}' || true"
test -d res/images || mkdir -p ../dogs
test -d res/images || cd ../dogs
test -d res/images || curl  --progress-bar http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n02084071 | tail -n 50 | tr -d '\r' | xargs -L1 -P 5 -I {} sh -c "curl --connect-timeout 10 --fail --silent --progress-bar --remote-name '{}' || true"
test -d res/images || cd ../../..
test -d res/images || mv res/download res/images
