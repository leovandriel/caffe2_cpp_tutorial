#ifdef WITH_CUDA

#include "caffe2/core/common_cudnn.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/core/types.h"
#include "operator/affine_scale_op.h"
#include "util/math.h"

namespace caffe2 {

template <typename T>
class CuDNNAffineScaleOp final : public Operator<CUDAContext> {
 public:
  explicit CuDNNAffineScaleOp(const OperatorDef& def, Workspace* ws)
      : Operator<CUDAContext>(def, ws),
        inverse_(OperatorBase::GetSingleArgument<int>("inverse", 0)) {}

  bool RunOnDevice() override {
    auto& X = Input(0);
    auto& M = Input(1);
    auto& S = Input(2);
    auto* Y = Output(0);
    Y->ResizeLike(X);
    get_affine_scale_tensor(X, M, S, *Y, inverse_);
    return true;
  }

 protected:
  int inverse_;
};


template <typename T>
class CuDNNAffineScaleGradientOp final : public Operator<CUDAContext> {
 public:
  explicit CuDNNAffineScaleGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<CUDAContext>(def, ws),
        inverse_(OperatorBase::GetSingleArgument<int>("inverse", 0)) {}

  bool RunOnDevice() override {
    auto& X = Input(0);
    auto& S = Input(1);
    auto& dY = Input(2);
    auto* dX = Output(0);
    dX->ResizeLike(X);
    set_affine_scale_tensor(*dX, S, dY, inverse_);
    return true;
  }

 protected:
  int inverse_;
};

namespace {

REGISTER_CUDNN_OPERATOR(AffineScale, CuDNNAffineScaleOp<float>);
REGISTER_CUDNN_OPERATOR(AffineScaleGradient, CuDNNAffineScaleGradientOp<float>);

}  // namespace

}  // namespace caffe2

#endif
