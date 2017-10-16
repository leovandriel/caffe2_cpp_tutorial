#include "caffe2/operator/mean_stdev_op.h"

namespace caffe2 {

template <typename C>
void get_mean_stdev_tensor(const Tensor<C>& tensor, Tensor<C>& mean,
                           Tensor<C>& stdev) {
  auto data = tensor.template data<float>();
  auto size = tensor.size() / tensor.dim(0);
  auto mean_data = mean.template mutable_data<float>();
  auto stdev_data = stdev.template mutable_data<float>();
  for (auto e = data + tensor.size(); data != e;
       data += size, mean_data++, stdev_data++) {
    auto sum = 0.f;
    for (auto d = data, e = data + size; d != e; d++) {
      sum += *d;
    }
    auto mean = sum / size;
    auto sq_sum = 0.f;
    for (auto d = data, e = data + size; d != e; d++) {
      sq_sum += (*d - mean) * (*d - mean);
    }
    auto stdev = sqrt(sq_sum / size);
    *mean_data = mean;
    *stdev_data = stdev;
  }
}

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

}  // namespace caffe2
