#ifdef WITH_CUDA

#include "caffe2/core/common_cudnn.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/core/types.h"
#include "operator/back_mean_op.h"
#include "util/math.h"

namespace caffe2 {

template <typename T>
class CuDNNBackMeanOp final : public Operator<CUDAContext> {
 public:
  explicit CuDNNBackMeanOp(const OperatorDef& def, Workspace* ws)
      : Operator<CUDAContext>(def, ws),
        count_(OperatorBase::GetSingleArgument<int>("count", 1)) {}

  bool RunOnDevice() override {
    auto& X = Input(0);
    auto* Y = Output(0);
    get_back_mean_tensor(X, *Y, count_);
    return true;
  }

 protected:
  int count_;
};


template <typename T>
class CuDNNBackMeanGradientOp final : public Operator<CUDAContext> {
 public:
  explicit CuDNNBackMeanGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<CUDAContext>(def, ws),
        count_(OperatorBase::GetSingleArgument<int>("count", 1)) {}

  bool RunOnDevice() override {
    auto& X = Input(0);
    auto& dY = Input(1);
    auto* dX = Output(0);
    dX->ResizeLike(X);
    set_back_mean_tensor(*dX, dY, count_);
    return true;
  }

 protected:
  int count_;
};

namespace {

REGISTER_CUDNN_OPERATOR(BackMean, CuDNNBackMeanOp<float>);
REGISTER_CUDNN_OPERATOR(BackMeanGradient, CuDNNBackMeanGradientOp<float>);

}  // namespace

}  // namespace caffe2

#endif
