#ifndef UTIL_TENSOR_H
#define UTIL_TENSOR_H

#include <caffe2/core/tensor.h>

namespace caffe2 {

class TensorUtil {
 public:
  TensorUtil(Tensor<CPUContext>& tensor) : tensor_(tensor) {}

  void ShowImages(const std::string& name, float scale = 1.0, float mean = 128);
  void ShowImage(const std::string& title, int index, float scale = 1.0,
                 float mean = 128);
  void WriteImages(const std::string& name, float mean = 128,
                   bool lossy = false, int index = 0);
  void WriteImage(const std::string& name, int index, float mean = 128,
                  bool lossy = false);
  TensorCPU ScaleImageTensor(int width, int height);
  void ReadImages(const std::vector<std::string>& filenames, int size,
                  std::vector<int>& indices, float mean = 128,
                  TensorProto::DataType type = TensorProto_DataType_FLOAT);
  void ReadImage(const std::string& filename, int size);
  void Print(const std::string& name = "", int max = 100);

 protected:
  Tensor<CPUContext>& tensor_;
};

}  // namespace caffe2

#endif  // UTIL_TENSOR_H
