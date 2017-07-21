#ifndef UTIL_TENSOR_H
#define UTIL_TENSOR_H

#include "caffe2/core/tensor.h"

namespace caffe2 {

class TensorUtil {
 public:
  TensorUtil(const Tensor<CPUContext>& tensor):
    tensor_(tensor) {}

  void ShowImages(int width, int height, const std::string& name, float mean = 128);
  void ShowImage(int width, int height, int index, const std::string& name, int offset = 0, int wait = 1, float mean = 128);
  void WriteImages(const std::string& name, float mean = 128);
  TensorCPU ScaleImageTensor(int width, int height);

 protected:
   const Tensor<CPUContext>& tensor_;
};

}  // namespace caffe2

#endif  // UTIL_TENSOR_H
