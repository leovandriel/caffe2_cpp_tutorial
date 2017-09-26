#include <caffe2/operator/zero_one_op.h>

namespace caffe2 {

template <>
bool ZeroOneOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(PREDICTION);
  auto& label = Input(LABEL);
  DCHECK_EQ(X.ndim(), 2);
  int N = X.dim32(0);
  int D = X.dim32(1);
  DCHECK_EQ(label.ndim(), 1);
  DCHECK_EQ(label.dim32(0), N);
  const auto* Xdata = X.data<float>();
  const auto* labelData = label.data<int>();

  for (int i = 0; i < N; ++i) {
    auto label_i = labelData[i];
    auto label_pred = Xdata[i * D + label_i];
    auto correct = true;
    for (int j = 0; j < D; ++j) {
      auto pred = Xdata[i * D + j];
      if ((pred > label_pred) || (pred == label_pred && j < label_i)) {
        correct = false;
        break;
      }
    }
    std::cout << correct;
  }
  std::cout << std::endl;

  return true;
}

REGISTER_CPU_OPERATOR(ZeroOne, ZeroOneOp<float, CPUContext>);

OPERATOR_SCHEMA(ZeroOne)
    .NumInputs(2)
    .NumOutputs(0)
    .ScalarType(TensorProto::FLOAT)
    .SetDoc("Display classification success of batch with zeros and ones.")
    .Input(0, "predictions",
           "2-D tensor (Tensor<float>) of size "
           "(num_batches x num_classes) containing scores")
    .Input(1, "labels",
           "1-D tensor (Tensor<int>) of size (num_batches) having "
           "the indices of true labels");

SHOULD_NOT_DO_GRADIENT(ZeroOne);

}  // namespace caffe2
