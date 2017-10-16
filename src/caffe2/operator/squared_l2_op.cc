#include "caffe2/operator/squared_l2_op.h"

namespace caffe2 {

template <>
bool SquaredL2Op<float, CPUContext>::RunOnDevice() {
  auto& X = Input(0);
  auto* Y = Output(0);
  int N = X.ndim() > 0 ? X.dim32(0) : 1;
  Y->Resize(N);
  int D = N > 0 ? X.size() / N : 0;
  float* Y_data = Y->mutable_data<float>();
  const float* X_data = X.data<float>();
  for (int i = 0; i < N; ++i) {
    float Xscale;
    math::Dot<float, CPUContext>(D, X_data + i * D, X_data + i * D, &Xscale,
                                 &context_);
    Y_data[i] = Xscale;
  }
  return true;
}

template <>
bool SquaredL2GradientOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(0);
  auto& dY = Input(1);
  auto* dX = Output(0);
  int N = X.ndim() > 0 ? X.dim32(0) : 1;
  int D = N > 0 ? X.size() / N : 0;
  CAFFE_ENFORCE_EQ(dY.ndim(), 1);
  CAFFE_ENFORCE_EQ(dY.dim32(0), N);
  dX->ResizeLike(X);
  for (int i = 0; i < N; ++i) {
    math::Scale<float, CPUContext>(
        D, dY.template data<float>() + i, X.template data<float>() + i * D,
        dX->template mutable_data<float>() + i * D, &context_);
  }
  return true;
}

REGISTER_CPU_OPERATOR(SquaredL2, SquaredL2Op<float, CPUContext>);
REGISTER_CPU_OPERATOR(SquaredL2Gradient,
                      SquaredL2GradientOp<float, CPUContext>);

OPERATOR_SCHEMA(SquaredL2)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
The squared L2 distance to zero.
)DOC")
    .Input(0, "input", "The input data as N-D Tensor<float>.")
    .Output(0, "output", "The mean values in a N-D Tensor.");

OPERATOR_SCHEMA(SquaredL2Gradient).NumInputs(2).NumOutputs(1);

class GetSquaredL2Gradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(def_.type() + "Gradient", "",
                             vector<string>{I(0), GO(0)},
                             vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(SquaredL2, GetSquaredL2Gradient);

}  // namespace caffe2
