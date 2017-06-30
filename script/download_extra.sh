#!/bin/bash

set -e

mkdir -p res

test -f res/alexnet_init_net.pb || echo "downloading AlexNet model (2)"
test -f res/alexnet_predict_net.pb || curl --progress-bar --location --output res/alexnet_predict_net.pb https://s3.amazonaws.com/caffe2/models/bvlc_alexnet/predict_net.pb
test -f res/alexnet_init_net.pb || curl --progress-bar --location --output res/alexnet_init_net.pb https://s3.amazonaws.com/caffe2/models/bvlc_alexnet/init_net.pb

test -f res/googlenet_init_net.pb || echo "downloading GoogleNet model (2)"
test -f res/googlenet_predict_net.pb || curl --progress-bar --location --output res/googlenet_predict_net.pb https://s3.amazonaws.com/caffe2/models/bvlc_googlenet/predict_net.pb
test -f res/googlenet_init_net.pb || curl --progress-bar --location --output res/googlenet_init_net.pb https://s3.amazonaws.com/caffe2/models/bvlc_googlenet/init_net.pb

test -f res/squeezenet_init_net.pb || echo "downloading SqueezeNet model (2)"
test -f res/squeezenet_predict_net.pb || curl --progress-bar --location --output res/squeezenet_predict_net.pb https://s3.amazonaws.com/caffe2/models/squeezenet/predict_net.pb
test -f res/squeezenet_init_net.pb || curl --progress-bar --location --output res/squeezenet_init_net.pb https://s3.amazonaws.com/caffe2/models/squeezenet/init_net.pb

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

