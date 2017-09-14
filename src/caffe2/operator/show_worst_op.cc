#include "caffe2/operator/show_worst_op.h"

#include "caffe2/util/tensor.h"

#ifdef WITH_CUDA
#include "caffe2/core/common_cudnn.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/core/types.h"
#endif

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

namespace caffe2 {

void show_worst(const TensorCPU &X, const TensorCPU &label,
                const TensorCPU &image) {
  // auto iter = -1;
  // if (InputSize() > 3) {
  //   iter = Input(ITER).data<int64_t>()[0];
  //   if (iter % 10) {
  //     return true;
  //   }
  // }
  DCHECK_EQ(X.ndim(), 2);
  int N = X.dim32(0);
  int D = X.dim32(1);
  int CHW = image.size() / image.dim32(0);
  DCHECK_EQ(label.ndim(), 1);
  DCHECK_EQ(label.dim32(0), N);
  const auto *Xdata = X.data<float>();
  const auto *labelData = label.data<int>();
  const auto *imageData = image.data<float>();

  int pos_i = -1;
  int pos_label = -1;
  float pos_pred = -1;
  int neg_i = -1;
  int neg_label = -1;
  float neg_pred = -1;
  for (int i = 0; i < N; ++i) {
    auto label_i = labelData[i];
    auto label_pred = Xdata[i * D + label_i];
    int best_j = -1;
    float best_pred = -1;
    for (int j = 0; j < D; ++j) {
      auto pred = Xdata[i * D + j];
      if (best_j < 0 || best_pred < pred) {
        best_pred = pred;
        best_j = j;
      }
    }
    auto correct = (best_j == label_i);
    if (correct && (pos_i < 0 || pos_pred > label_pred)) {
      pos_pred = label_pred;
      pos_label = best_j;
      pos_i = i;
    }
    if (!correct && (neg_i < 0 || neg_pred < label_pred)) {
      neg_pred = label_pred;
      neg_label = best_j;
      neg_i = i;
    }
  }

  TensorCPU t(image);
  if (pos_i >= 0) {
    TensorUtil(t).ShowImage("worst_pos", pos_i, 256, 0);
  }
  if (neg_i >= 0) {
    TensorUtil(t).ShowImage("worst_neg", neg_i, 256, 0);
  }
}

template <>
bool ShowWorstOp<float, CPUContext>::RunOnDevice() {
  show_worst(Input(PREDICTION), Input(LABEL), Input(DATA));
  return true;
}

#ifdef WITH_CUDA
template <>
bool ShowWorstOp<float, CUDAContext>::RunOnDevice() {
  show_worst(TensorCPU(Input(PREDICTION)), TensorCPU(Input(LABEL)),
             TensorCPU(Input(DATA)));
  return true;
}
#endif

REGISTER_CPU_OPERATOR(ShowWorst, ShowWorstOp<float, CPUContext>);

#ifdef WITH_CUDA
REGISTER_CUDA_OPERATOR(ShowWorst, ShowWorstOp<float, CUDAContext>);
#endif

OPERATOR_SCHEMA(ShowWorst)
    .NumInputs(3, 4)
    .NumOutputs(0)
    .ScalarType(TensorProto::FLOAT)
    .SetDoc("Show worst correct and incorrect classification within the batch.")
    .Input(0, "predictions",
           "2-D tensor (Tensor<float>) of size "
           "(num_batches x num_classes) containing scores")
    .Input(1, "labels",
           "1-D tensor (Tensor<int>) of size (num_batches) having "
           "the indices of true labels")
    .Input(2, "data",
           "4-D tensor (Tensor<float>) of size "
           "(N x C x H x Wwidth), where N is the batch size, C is the number "
           "of channels, and"
           " H and W are the height and width. Note that this is for the NCHW "
           "usage. On "
           "the other hand, the NHWC Op has a different set of dimension "
           "constraints.");
// .Input(3, "iter", "1-D tensor (Tensor<int>) of size (1) having training
// iteraton");

SHOULD_NOT_DO_GRADIENT(ShowWorst);

}  // namespace caffe2
