#ifndef CUDA_H
#define CUDA_H

#ifdef WITH_CUDA
  #include "caffe2/core/context_gpu.h"
#endif

namespace caffe2 {

bool setupCUDA() {
  DeviceOption option;
  option.set_device_type(CUDA);
#ifdef WITH_CUDA
  new CUDAContext(option);
  return true;
#else
  return false;
#endif
}

void addCUDNN(NetDef &model) {
  DeviceOption option;
  option.set_device_type(CUDA);
  *model.mutable_device_option() = option;
  for (auto op: model.op()) {
      op.set_engine("CUDNN");
      op.mutable_device_option()->set_device_type(CUDA);
  }
}

}  // namespace caffe2

#endif  // CUDA_H
