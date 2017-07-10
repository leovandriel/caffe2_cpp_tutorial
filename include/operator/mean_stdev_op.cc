#include "operator/mean_stdev_op.h"
#include "util/math.h"

namespace caffe2 {

template <>
bool MeanStdevOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(0);
  auto* M = Output(0);
  auto* S = Output(1);
  M->Resize(X.dim(0));
  S->Resize(X.dim(0));
  get_mean_stdev_tensor(X, *M, *S);
  return true;
}

namespace {

REGISTER_CPU_OPERATOR(MeanStdev, MeanStdevOp<float, CPUContext>);

OPERATOR_SCHEMA(MeanStdev)
  .NumInputs(1)
  .NumOutputs(2)
  .SetDoc(R"DOC(
The operator computes the mean and stdev over the batch.
)DOC")
  .Input(0, "input", "The input data as N-D Tensor<float>.")
  .Output(0, "mean", "The mean in Tensor of batch size.")
  .Output(1, "stdev", "The stdev in Tensor of batch size.");

SHOULD_NOT_DO_GRADIENT(MeanStdev);

}  // namespace

}  // namespace caffe2
