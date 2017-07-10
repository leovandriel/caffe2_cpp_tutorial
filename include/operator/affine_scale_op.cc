#include "operator/affine_scale_op.h"
#include "util/math.h"

namespace caffe2 {

template <>
bool AffineScaleOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(0);
  auto& M = Input(1);
  auto& S = Input(2);
  auto* Y = Output(0);
  Y->ResizeLike(X);
  get_affine_scale_tensor(X, M, S, *Y, inverse_);
  return true;
}

template <>
bool AffineScaleGradientOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(0);
  auto& S = Input(1);
  auto& dY = Input(2);
  auto* dX = Output(0);
  dX->ResizeLike(X);
  set_affine_scale_tensor(*dX, S, dY, inverse_);
  return true;
}

namespace {

REGISTER_CPU_OPERATOR(AffineScale, AffineScaleOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(AffineScaleGradient, AffineScaleGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(AffineScale)
  .NumInputs(3)
  .NumOutputs(1)
  .AllowInplace({{0, 0}})
  .IdenticalTypeAndShape()
  .SetDoc(R"DOC(
The operator affine transforms values per batch item.
)DOC")
  .Arg("inverse", "apply inverse of affine transform")
  .Input(0, "input", "The input data as N-D Tensor<float>.")
  .Input(1, "mean", "The mean values as 1-D Tensor<float> of batch size.")
  .Input(2, "scale", "The scale values as 1-D Tensor<float> of batch size.")
  .Output(0, "output", "Scaled N-D Tensor<float>.");

OPERATOR_SCHEMA(AffineScaleGradient).NumInputs(4).NumOutputs(1).AllowInplace({{0, 0}});

class GetAffineScaleGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        def_.type() + "Gradient", "",
        vector<string>{I(0), I(2), GO(0)},
        vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(AffineScale, GetAffineScaleGradient);

}  // namespace

}  // namespace caffe2
