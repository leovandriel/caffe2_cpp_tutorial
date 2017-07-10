#ifdef WITH_CUDA

#include "caffe2/core/common_cudnn.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/core/types.h"
#include "operator/diagonal_op.h"
#include "util/math.h"

namespace caffe2 {

template <typename T>
class CuDNNDiagonalOp final : public Operator<CUDAContext> {
 public:
  explicit CuDNNDiagonalOp(const OperatorDef& def, Workspace* ws)
      : Operator<CUDAContext>(def, ws),
        offset_(OperatorBase::GetRepeatedArgument<TIndex>("offset")) {}

  bool RunOnDevice() override {
    auto& X = Input(0);
    auto* Y = Output(0);
    get_diagonal_tensor(X, *Y, offset_);
    return true;
  }

 protected:
  std::vector<TIndex> offset_;
};


template <typename T>
class CuDNNDiagonalGradientOp final : public Operator<CUDAContext> {
 public:
  explicit CuDNNDiagonalGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<CUDAContext>(def, ws),
        offset_(OperatorBase::GetRepeatedArgument<TIndex>("offset")) {}

  bool RunOnDevice() override {
    auto& X = Input(0);
    auto& dY = Input(1);
    auto* dX = Output(0);
    dX->ResizeLike(X);
    set_diagonal_tensor(*dX, dY, offset_);
    return true;
  }

 protected:
  std::vector<TIndex> offset_;
};

namespace {

REGISTER_CUDNN_OPERATOR(Diagonal, CuDNNDiagonalOp<float>);
REGISTER_CUDNN_OPERATOR(DiagonalGradient, CuDNNDiagonalGradientOp<float>);

}  // namespace

}  // namespace caffe2

#endif
