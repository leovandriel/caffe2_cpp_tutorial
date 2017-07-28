#include "caffe2/operator/affine_scale_op.h"

namespace caffe2 {

template<typename C>
void get_affine_scale_tensor(const Tensor<C> &tensor, const Tensor<C> &mean, const Tensor<C> &scale, Tensor<C> &transformed, bool inverse = false) {
  auto data = tensor.template data<float>();
  auto size = tensor.size() / tensor.dim(0);
  auto mean_data = mean.template data<float>();
  auto scale_data = scale.template data<float>();
  auto transformed_data = transformed.template mutable_data<float>();
  for (auto e = data + tensor.size(); data != e; mean_data++, scale_data++) {
    for (auto f = data + size; data != f; data++, transformed_data++) {
      if (inverse) {
        *transformed_data = (*data - *mean_data) / (*scale_data + 1e-8);
      } else {
        *transformed_data = *data * *scale_data + *mean_data;
      }
    }
  }
}

template<typename C>
void set_affine_scale_tensor(Tensor<C> &tensor, const Tensor<C> &scale, const Tensor<C> &transformed, bool inverse = false) {
  auto data = tensor.template mutable_data<float>();
  auto size = tensor.size() / tensor.dim(0);
  auto scale_data = scale.template data<float>();
  auto transformed_data = transformed.template data<float>();
  for (auto e = data + tensor.size(); data != e; scale_data++) {
    for (auto f = data + size; data != f; data++, transformed_data++) {
      if (inverse) {
        *data = *transformed_data / (*scale_data + 1e-8);
      } else {
        *data = *transformed_data * *scale_data;
      }
    }
  }
}

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
