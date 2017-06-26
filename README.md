# Caffe2 C++ Tutorial

*C++ transcripts of the Caffe2 Python tutorials*


## About

Caffe2 has a strong C++ core but most tutorials only cover the outer Python layer of the framework. This project aims to provide example code written in C++, complementary to the Python documentation and tutorials. It covers verbatim transcriptions of most of the Python tutorials and other example applications. Some higher level tools, like brewing models and adding gradient operations are currently not available in Caffe2 C++, which makes these transcriptions more verbose.

Check out the actual tutorials at [https://caffe2.ai/docs/tutorials.html](https://caffe2.ai/docs/tutorials.html).


## Build

1. Install dependencies

    Install the dependencies CMake, leveldb and OpenCV. If you're on macOS, use Homebrew:

        brew install cmake leveldb opencv

2. Install Caffe2

    Follow the Caffe2 installation instructions: [https://caffe2.ai/docs/getting-started.html](https://caffe2.ai/docs/getting-started.html)

3. Build using Make

    This project uses CMake. However easiest way to just build the whole thing is:

        make

    Internally it creates a `build` folder and runs CMake from there. This also downloads the resources that are required for running some of the tutorials.

Note: sources are developed and tested on macOS, but should be fairly OS-agnostic.


## Usage

Specific tutorials can be compiled and run with:

    make && ./bin/<name-of-tutorial>

For example, to run the MNIST tutorial, run:

    make && ./bin/mnist

The following tutorials have been transcribed:

* `intro`: [Intro Tutorial](https://caffe2.ai/docs/intro-tutorial.html)
* `toy`: [Toy Regression](https://caffe2.ai/docs/tutorial-toy-regression.html)
* `pretrained`: [Loading Pre-Trained Models](https://caffe2.ai/docs/tutorial-loading-pre-trained-models.html)
* `mnist`: [MNIST - Create a CNN from Scratch](https://caffe2.ai/docs/tutorial-MNIST.html)

There's also examples of other common architectures:

* `alexnet`: [AlexNet](https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet)
* `googlenet`: [GoogleNet](https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet)
* `squeezenet`: [SqueezeNet](https://github.com/DeepScale/SqueezeNet)
* `vgg16`: [VGG Team](http://www.robots.ox.ac.uk/~vgg/research/very_deep/)


## Troubleshooting

See [http://rpg.ifi.uzh.ch/docs/glog.html](http://rpg.ifi.uzh.ch/docs/glog.html) for more info on logging. Try running the tools and examples with `--logtostderr=1`, `--caffe2_log_level=1`, and `--v=1`.
