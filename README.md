# Caffe2 C++ Tutorial

*C++ transcripts of the Caffe2 Python tutorials*

## About

See [https://caffe2.ai/docs/tutorials.html](https://caffe2.ai/docs/tutorials.html).

## Build

1. Install dependencies

        brew install leveldb opencv

2. Install Caffe2

    See [https://caffe2.ai/docs/getting-started.html](https://caffe2.ai/docs/getting-started.html).

3. Build using Make (or CMake)

        make

Sources are developed and tested on macOS, but should be fairly OS-agnostic.

## Usage

The following tutorials have been transcribed:

* Intro

    See [https://caffe2.ai/docs/intro-tutorial.html](https://caffe2.ai/docs/intro-tutorial.html).

        make && ./bin/intro

* Toy Regression

    See [https://caffe2.ai/docs/tutorial-toy-regression.html](https://caffe2.ai/docs/tutorial-toy-regression.html).

        make && ./bin/toy

* Squeezenet

    See [https://caffe2.ai/docs/zoo.html](https://caffe2.ai/docs/zoo.html) and [https://caffe2.ai/docs/tutorial-loading-pre-trained-models.html](https://caffe2.ai/docs/tutorial-loading-pre-trained-models.html).

        make && ./bin/squeeze

* MNIST

    See [https://caffe2.ai/docs/tutorial-MNIST.html](https://caffe2.ai/docs/tutorial-MNIST.html).

        make && ./bin/mnist

See [http://rpg.ifi.uzh.ch/docs/glog.html](http://rpg.ifi.uzh.ch/docs/glog.html) for more info on logging. Try `--logtostderr=1`, `--caffe2_log_level=0`, and `--v=1`.
