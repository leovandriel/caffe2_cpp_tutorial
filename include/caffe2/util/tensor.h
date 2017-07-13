#ifndef UTIL_TENSOR_H
#define UTIL_TENSOR_H

#include "caffe2/core/tensor.h"

#include "opencv2/opencv.hpp"

namespace caffe2 {

class TensorUtil {
 public:
  TensorUtil(const Tensor<CPUContext>& tensor):
    tensor_(tensor) {}

  cv::Mat toImage(int index, float mean = 128);
  void showImages(int width, int height, const std::string& name = "default", float mean = 128);
  void showImage(int width, int height, int index, const std::string& name = "default", int offset = 0, float mean = 128);
  void writeImages(const std::string& name, float mean = 128);

 protected:
   const Tensor<CPUContext>& tensor_;
};

}  // namespace caffe2

#endif  // UTIL_TENSOR_H
