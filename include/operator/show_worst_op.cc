#include "operator/show_worst_op.h"

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

// #include "util/print.h"

namespace caffe2 {

template <>
bool ShowWorstOp<float, CPUContext>::RunOnDevice() {
  auto &X = Input(PREDICTION);
  auto &label = Input(LABEL);
  auto &image = Input(DATA);
  auto iter = -1;
  if (InputSize() > 3) {
    iter = Input(ITER).data<int64_t>()[0];
    if (iter % 10) {
      return true;
    }
  }
  DCHECK_EQ(X.ndim(), 2);
  int N = X.dim32(0);
  int D = X.dim32(1);
  int CHW = image.dim32(1) * image.dim32(2) * image.dim32(3);
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

  float scale = 200.f / image.dim32(3);
  if (pos_i > 0) {
    auto data = imageData + (pos_i * image.dim32(1) * image.dim32(2) * image.dim32(3));
    vector<cv::Mat> channels(image.dim32(1));
    for (auto &j: channels) {
      j = cv::Mat(image.dim32(2), image.dim32(3), CV_32F, (void *)data);
      data += (image.dim32(2) * image.dim32(3));
    }
    cv::Mat mat;
    cv::merge(channels, mat);
    mat.convertTo(mat, CV_8UC3, 1.0, 128);

    cv::resize(mat, mat, cv::Size(), scale, scale, cv::INTER_NEAREST);
    cv::namedWindow("pos", cv::WINDOW_AUTOSIZE);
    // cv::setWindowTitle("pos", "worst correct, pred: " + std::to_string((int)(100 * pos_pred)) + "%  label: " + std::to_string(pos_label) + "  iter: " + std::to_string(iter));
    cv::imshow("pos", mat);
  }
  if (neg_i > 0) {
    auto data = imageData + (neg_i * image.dim32(1) * image.dim32(2) * image.dim32(3));
    vector<cv::Mat> channels(image.dim32(1));
    for (auto &j: channels) {
      j = cv::Mat(image.dim32(2), image.dim32(3), CV_32F, (void *)data);
      data += (image.dim32(2) * image.dim32(3));
    }
    cv::Mat mat;
    cv::merge(channels, mat);
    mat.convertTo(mat, CV_8UC3, 1.0, 128);

    cv::resize(mat, mat, cv::Size(), scale, scale, cv::INTER_NEAREST);
    cv::namedWindow("neg", cv::WINDOW_AUTOSIZE);
    // cv::setWindowTitle("neg", "worst incorrect, pred: " + std::to_string((int)(100 * neg_pred)) + "%  label: " + std::to_string(neg_label) + "  iter: " + std::to_string(iter));
    cv::moveWindow("neg", mat.cols, 0);
    cv::imshow("neg", mat);
  }
  cv::waitKey(1000);

  return true;
}

namespace {

REGISTER_CPU_OPERATOR(ShowWorst, ShowWorstOp<float, CPUContext>);

OPERATOR_SCHEMA(ShowWorst)
  .NumInputs(3, 4)
  .NumOutputs(0)
  .ScalarType(TensorProto::FLOAT)
  .SetDoc("Show worst correct and incorrect classification within the batch.")
  .Input(0, "predictions", "2-D tensor (Tensor<float>) of size "
         "(num_batches x num_classes) containing scores")
  .Input(1, "labels", "1-D tensor (Tensor<int>) of size (num_batches) having "
        "the indices of true labels")
  .Input(2, "data", "4-D tensor (Tensor<float>) of size "
        "(N x C x H x Wwidth), where N is the batch size, C is the number of channels, and"
        " H and W are the height and width. Note that this is for the NCHW usage. On "
        "the other hand, the NHWC Op has a different set of dimension constraints.")
  .Input(3, "iter", "1-D tensor (Tensor<int>) of size (1) having training iteraton");

SHOULD_NOT_DO_GRADIENT(ShowWorst);

}  // namespace

}  // namespace caffe2
