#!/bin/bash

set -e

mkdir -p res

test -f res/alexnet_init_net.pb || echo "downloading AlexNet model (2)"
test -f res/alexnet_predict_net.pb || curl --progress-bar --location --output res/alexnet_predict_net.pb https://github.com/leonardvandriel/caffe2_models/raw/master/model/alexnet_predict_net.pb
test -f res/alexnet_init_net.pb || curl --progress-bar --location --output res/alexnet_init_net.pb https://github.com/leonardvandriel/caffe2_models/raw/master/model/alexnet_init_net.pb

test -f res/googlenet_init_net.pb || echo "downloading GoogleNet model (2)"
test -f res/googlenet_predict_net.pb || curl --progress-bar --location --output res/googlenet_predict_net.pb https://github.com/leonardvandriel/caffe2_models/raw/master/model/googlenet_predict_net.pb
test -f res/googlenet_init_net.pb || curl --progress-bar --location --output res/googlenet_init_net.pb https://github.com/leonardvandriel/caffe2_models/raw/master/model/googlenet_init_net.pb

test -f res/squeezenet_init_net.pb || echo "downloading SqueezeNet model (2)"
test -f res/squeezenet_predict_net.pb || curl --progress-bar --location --output res/squeezenet_predict_net.pb https://github.com/caffe2/models/raw/master/squeezenet/predict_net.pb
test -f res/squeezenet_init_net.pb || curl --progress-bar --location --output res/squeezenet_init_net.pb https://github.com/caffe2/models/raw/master/squeezenet/init_net.pb

test -f res/vgg16_init_net.pb || echo "downloading VGG16 model (2)"
test -f res/vgg16_predict_net.pb || curl --progress-bar --location --output res/vgg16_predict_net.pb https://github.com/leonardvandriel/caffe2_models/raw/master/model/vgg16_predict_net.pb
test -f res/vgg16_init_net.pb || curl --progress-bar --location --output res/vgg16_init_net.pb https://github.com/leonardvandriel/caffe2_models/raw/master/model/vgg16_init_net.pb

test -f res/vgg19_init_net.pb || echo "downloading VGG19 model (2)"
test -f res/vgg19_predict_net.pb || curl --progress-bar --location --output res/vgg19_predict_net.pb https://github.com/leonardvandriel/caffe2_models/raw/master/model/vgg19_predict_net.pb
test -f res/vgg19_init_net.pb || curl --progress-bar --location --output res/vgg19_init_net.pb https://github.com/leonardvandriel/caffe2_models/raw/master/model/vgg19_init_net.pb

test -f res/resnet50_init_net.pb || echo "downloading ResNet-50 model (2)"
test -f res/resnet50_predict_net.pb || curl --progress-bar --location --output res/resnet50_predict_net.pb https://github.com/leonardvandriel/caffe2_models/raw/master/model/resnet50_predict_net.pb
test -f res/resnet50_init_net.pb || curl --progress-bar --location --output res/resnet50_init_net.pb https://github.com/leonardvandriel/caffe2_models/raw/master/model/resnet50_init_net.pb

test -f res/resnet101_init_net.pb || echo "downloading ResNet-101 model (2)"
test -f res/resnet101_predict_net.pb || curl --progress-bar --location --output res/resnet101_predict_net.pb https://github.com/leonardvandriel/caffe2_models/raw/master/model/resnet101_predict_net.pb
test -f res/resnet101_init_net.pb || curl --progress-bar --location --output res/resnet101_init_net.pb https://github.com/leonardvandriel/caffe2_models/raw/master/model/resnet101_init_net.pb

test -f res/resnet152_init_net.pb || echo "downloading ResNet-152 model (2)"
test -f res/resnet152_predict_net.pb || curl --progress-bar --location --output res/resnet152_predict_net.pb https://github.com/leonardvandriel/caffe2_models/raw/master/model/resnet152_predict_net.pb
test -f res/resnet152_init_net.pb || curl --progress-bar --location --output res/resnet152_init_net.pb https://github.com/leonardvandriel/caffe2_models/raw/master/model/resnet152_init_net.pb

test -f res/dickens_flat.txt || echo "downloading RNN and LSTM test data (10)"
test -f res/dickens_flat.txt || echo "" > res/dickens.txt
test -f res/dickens_flat.txt || curl --progress-bar https://www.gutenberg.org/files/98/98-0.txt >> res/dickens.txt
test -f res/dickens_flat.txt || curl --progress-bar https://www.gutenberg.org/files/1400/1400-0.txt >> res/dickens.txt
test -f res/dickens_flat.txt || curl --progress-bar https://www.gutenberg.org/files/766/766-0.txt >> res/dickens.txt
test -f res/dickens_flat.txt || curl --progress-bar https://www.gutenberg.org/files/786/786-0.txt >> res/dickens.txt
test -f res/dickens_flat.txt || curl --progress-bar https://www.gutenberg.org/files/580/580-0.txt >> res/dickens.txt
test -f res/dickens_flat.txt || curl --progress-bar https://www.gutenberg.org/files/883/883-0.txt >> res/dickens.txt
test -f res/dickens_flat.txt || curl --progress-bar https://www.gutenberg.org/files/967/967-0.txt >> res/dickens.txt
test -f res/dickens_flat.txt || curl --progress-bar https://www.gutenberg.org/files/963/963-0.txt >> res/dickens.txt
test -f res/dickens_flat.txt || curl --progress-bar https://www.gutenberg.org/files/700/700-0.txt >> res/dickens.txt
test -f res/dickens_flat.txt || curl --progress-bar https://www.gutenberg.org/files/821/821-0.txt >> res/dickens.txt
test -f res/dickens_flat.txt || cat res/dickens.txt | tr -cd 'A-Za-z0-9 \n!?,.:;()-' > res/dickens_strip.txt
test -f res/dickens_flat.txt || cat res/dickens.txt | tr -cd 'A-Za-z0-9 \n!?,.:;()-' | tr -d '\r' | tr '\n' '~' | sed -e 's/~~/^/g' | tr '^' '\n' | tr '~' ' ' > res/dickens_flat.txt
