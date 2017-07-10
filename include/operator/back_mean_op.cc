#include "operator/back_mean_op.h"
#include "util/math.h"

namespace caffe2 {

template <>
bool BackMeanOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(0);
  auto* Y = Output(0);
  get_back_mean_tensor(X, *Y, count_);
  return true;
}

template <>
bool BackMeanGradientOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(0);
  auto& dY = Input(1);
  auto* dX = Output(0);
  dX->ResizeLike(X);
  set_back_mean_tensor(*dX, dY, count_);
  return true;
}

namespace {

REGISTER_CPU_OPERATOR(BackMean, BackMeanOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(BackMeanGradient, BackMeanGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(BackMean)
  .NumInputs(1)
  .NumOutputs(1)
  .SetDoc(R"DOC(
The operator takes the mean values over the last dimensions.
)DOC")
  .Arg("count", "Number of dimensions to reduce")
  .Input(0, "input", "The input data as N-D Tensor<float>.")
  .Output(0, "output", "The mean values in a (N-count)-D Tensor.");

OPERATOR_SCHEMA(BackMeanGradient).NumInputs(2).NumOutputs(1);

class GetBackMeanGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        def_.type() + "Gradient", "",
        vector<string>{I(0), GO(0)},
        vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(BackMean, GetBackMeanGradient);

}  // namespace

}  // namespace caffe2
