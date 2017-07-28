#include "caffe2/operator/diagonal_op.h"

namespace caffe2 {

template <typename C>
void diag_step_size_tensor(const Tensor<C> &tensor, int *step_out,
                           int *size_out) {
  auto step = 0;
  auto size = tensor.dim(0);
  for (auto d : tensor.dims()) {
    step = step * d + 1;
    if (size > d) size = d;
  }
  if (step_out) *step_out = step;
  if (size_out) *size_out = size;
}

template <typename C>
int offset_in_tensor(const Tensor<C> &tensor,
                     const std::vector<TIndex> &offset) {
  auto off = 0, i = 0;
  for (auto d : tensor.dims()) {
    off = off * d + offset[i++];
  }
  return off;
}

template <typename C>
void get_diagonal_tensor(const Tensor<C> &tensor, Tensor<C> &diagonal,
                         const std::vector<TIndex> &offset) {
  auto off = offset_in_tensor(tensor, offset);
  auto step = 0, size = 0;
  diag_step_size_tensor(tensor, &step, &size);
  diagonal.Resize(size);
  auto data = tensor.template data<float>() + off;
  auto diagonal_data = diagonal.template mutable_data<float>();
  for (auto i = 0; i < size; i++, data += step, diagonal_data++) {
    *diagonal_data = *data;
  }
}

template <typename C>
void set_diagonal_tensor(Tensor<C> &tensor, const Tensor<C> &diagonal,
                         const std::vector<TIndex> &offset) {
  auto off = offset_in_tensor(tensor, offset);
  auto step = 0, size = 0;
  diag_step_size_tensor(tensor, &step, &size);
  auto data = tensor.template mutable_data<float>() + off;
  auto diagonal_data = diagonal.template data<float>();
  for (auto i = 0, s = (int)diagonal.size(); i < size;
       i++, data += step, diagonal_data++) {
    *data = i < s ? *diagonal_data : 0;
  }
}

template <>
bool DiagonalOp<float, CPUContext>::RunOnDevice() {
  auto &X = Input(0);
  auto *Y = Output(0);
  get_diagonal_tensor(X, *Y, offset_);
  return true;
}

template <>
bool DiagonalGradientOp<float, CPUContext>::RunOnDevice() {
  auto &X = Input(0);
  auto &dY = Input(1);
  auto *dX = Output(0);
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
    return SingleGradientDef(def_.type() + "Gradient", "",
                             vector<string>{I(0), GO(0)},
                             vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(Diagonal, GetDiagonalGradient);

}  // namespace

}  // namespace caffe2
