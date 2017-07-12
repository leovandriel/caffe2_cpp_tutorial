# Caffe2 C++ Tutorials and Examples

*C++ transcripts of the Caffe2 Python tutorials and other C++ example code.*


## About

Caffe2 has a strong C++ core but most tutorials only cover the outer Python layer of the framework. This project aims to provide example code written in C++, complementary to the Python documentation and tutorials. It covers verbatim transcriptions of most of the Python tutorials and other example applications. Some higher level tools, like brewing models and adding gradient operations are currently not available in Caffe2 C++, which makes these transcriptions more verbose.

Check out the actual tutorials at [https://caffe2.ai/docs/tutorials.html](https://caffe2.ai/docs/tutorials.html).


## Build

1. Install dependencies

    Install the dependencies CMake, leveldb and OpenCV. If you're on macOS, use Homebrew:

        brew install cmake leveldb opencv

    On Ubuntu:

        apt-get install cmake libleveldb-dev libopencv-dev libopencv-core-dev libopencv-highgui-dev

2. Install Caffe2

    Follow the Caffe2 installation instructions: [https://caffe2.ai/docs/getting-started.html](https://caffe2.ai/docs/getting-started.html)

3. Build using Make

    This project uses CMake. However easiest way to just build the whole thing is:

        make

    Internally it creates a `build` folder and runs CMake from there. This also downloads the resources that are required for running some of the tutorials.

Note: sources are developed and tested on macOS and Ubuntu.


## Tutorials

Specific tutorials can be compiled and run with:

    make && ./bin/<name-of-tutorial>

For example, to run the MNIST tutorial, run:

    make && ./bin/mnist

The following tutorials have been transcribed:

* `intro`: [Intro Tutorial](https://caffe2.ai/docs/intro-tutorial.html)
* `toy`: [Toy Regression](https://caffe2.ai/docs/tutorial-toy-regression.html)
* `pretrained`: [Loading Pre-Trained Models](https://caffe2.ai/docs/tutorial-loading-pre-trained-models.html)
* `mnist`: [MNIST - Create a CNN from Scratch](https://caffe2.ai/docs/tutorial-MNIST.html)

## ImageNet

There's also examples of other common architectures using ImageNet:

    make && ./bin/imagenet --model <name-of-model>

Names of available models:

* `alexnet`: [AlexNet](https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet)
* `googlenet`: [GoogleNet](https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet)
* `squeezenet`: [SqueezeNet](https://github.com/DeepScale/SqueezeNet)
* `vgg16` and `vgg19`: [VGG Team](http://www.robots.ox.ac.uk/~vgg/research/very_deep/)
* `resnet50`, `resnet101`, `resnet152`: [MSRA](https://github.com/KaimingHe/deep-residual-networks)

These models are taken from the [Model Zoo](https://github.com/caffe2/caffe2/wiki/Model-Zoo) and [Caffe2 Models](https://github.com/leonardvandriel/caffe2_models).

## Retrain

The above models are all trained on ImageNet data, which means they will only be able to classify ImageNet labels. However, they can be retrained on other image sets, so called transfer learning. If the image data similar characteristics, it's possible to get good results by only retraining the final layers.

First divide all images in subfolders with the label a folder name. Then to retrain the final layer of GoogleNet:

    make && ./bin/retrain --model googlenet --folder <image-folder> --layer pool5/7x7_s1

Or if you have more (GPU) power at your disposal retrain VGG16's final 2 layers:

    make && ./bin/retrain --model vgg16 --folder <image-folder> --layer fc6

See [DeCAF: A Deep Convolutional Activation Feature for Generic Visual Recognition](https://arxiv.org/pdf/1310.1531v1.pdf) for more information.

## Deep Dream

One way to inspect the weight parameters of a trained network is by using a technique called Deep Dream. This approach does backpropagation as usual, but instead of updating the weights, it updates the input data. By focussing on a specific channel in a specific layer, we can get an idea of what this particular part of the model was trained to recognize.

NB: this code is still a bit buggy and generated images are not representative of the original Deep Dream implementation.

The 139th channel in the `inception_4d/3x3` layer in GoogleNet:

    make && ./bin/dream --model googlenet --layer inception_4d/3x3_reduce --channel 139

Or if you have more (GPU) power at your disposal, the first channel in `conv3_1` layer in VGG16:

    make && ./bin/dream --model vgg16 --layer conv3_1

See [Inceptionism: Going Deeper into Neural Networks](https://research.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html) for more information.

## Troubleshooting

See [http://rpg.ifi.uzh.ch/docs/glog.html](http://rpg.ifi.uzh.ch/docs/glog.html) for more info on logging. Try running the tools and examples with `--logtostderr=1`, `--caffe2_log_level=1`, and `--v=1`.
