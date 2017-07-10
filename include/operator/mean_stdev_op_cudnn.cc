#ifdef WITH_CUDA

#include "caffe2/core/common_cudnn.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/core/types.h"
#include "operator/mean_stdev_op.h"
#include "util/math.h"

namespace caffe2 {

template <typename T>
class CuDNNMeanStdevOp final : public Operator<CUDAContext> {
 public:
  explicit CuDNNMeanStdevOp(const OperatorDef& def, Workspace* ws)
      : Operator<CUDAContext>(def, ws) {}

  bool RunOnDevice() override {
    auto& X = Input(0);
    auto* M = Output(0);
    auto* S = Output(1);
    M->Resize(X.dim(0));
    S->Resize(X.dim(0));
    get_mean_stdev_tensor(X, *M, *S);
    return true;
  }
};

namespace {

REGISTER_CUDNN_OPERATOR(MeanStdev, CuDNNMeanStdevOp<float>);

}  // namespace

}  // namespace caffe2

#endif
