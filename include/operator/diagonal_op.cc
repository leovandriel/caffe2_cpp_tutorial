#include "operator/diagonal_op.h"
#include "util/math.h"

namespace caffe2 {

template <>
bool DiagonalOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(0);
  auto* Y = Output(0);
  get_diagonal_tensor(X, *Y, offset_);
  return true;
}

template <>
bool DiagonalGradientOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(0);
  auto& dY = Input(1);
  auto* dX = Output(0);
  dX->ResizeLike(X);
  set_diagonal_tensor(*dX, dY, offset_);
  return true;
}

namespace {

REGISTER_CPU_OPERATOR(Diagonal, DiagonalOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(DiagonalGradient, DiagonalGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(Diagonal)
  .NumInputs(1)
  .NumOutputs(1)
  .SetDoc(R"DOC(
The operator takes the diagonal values from the input into the 1D output.
)DOC")
  .Arg("offset", "List of offset per dimension")
  .Input(0, "input", "The input data as n-D Tensor<float>.")
  .Output(0, "diagonal", "The diagonal of length miniumum over input dims.");

OPERATOR_SCHEMA(DiagonalGradient).NumInputs(2).NumOutputs(1);

class GetDiagonalGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        def_.type() + "Gradient", "",
        vector<string>{I(0), GO(0)},
        vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(Diagonal, GetDiagonalGradient);

}  // namespace

}  // namespace caffe2
