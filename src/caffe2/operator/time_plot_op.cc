#include "caffe2/operator/time_plot_op.h"

#include "caffe2/util/plot.h"
#include "caffe2/util/tensor.h"

#ifdef WITH_CUDA
#include "caffe2/core/common_cudnn.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/core/types.h"
#endif

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

namespace caffe2 {

void time_plot(const TensorCPU &X, const std::string &label_,
               const std::string &window_, const int &step_, const int &index_,
               int &step_count_, float &index_sum_, float &value_sum_) {
  step_count_++;
  value_sum_ += X.data<float>()[0];
  index_sum_ += index_;
  if (step_count_ >= step_) {
    auto &figure = PlotUtil::Shared(window_);
    figure.Get(label_).Set(PlotUtil::Line, PlotUtil::Color(label_));
    figure.Get(label_).Append(index_sum_ / step_count_,
                              value_sum_ / step_count_);
    figure.Show();
    value_sum_ = 0.f;
    index_sum_ = 0.f;
    step_count_ = 0;
  }
}

template <>
bool TimePlotOp<float, CPUContext>::RunOnDevice() {
  auto index = index_++;
  if (InputSize() > 1) {
    index = TensorCPU(Input(1)).data<int64_t>()[0];
  }
  time_plot(Input(0), label_, window_, step_, index, step_count_, index_sum_,
            value_sum_);
  return true;
}

#ifdef WITH_CUDA
template <>
bool TimePlotOp<float, CUDAContext>::RunOnDevice() {
  auto index = index_++;
  if (InputSize() > 1) {
    index = TensorCPU(Input(1)).data<int64_t>()[0];
  }
  time_plot(TensorCPU(Input(0)), label_, window_, step_, index, step_count_,
            index_sum_, value_sum_);
  return true;
}
#endif

REGISTER_CPU_OPERATOR(TimePlot, TimePlotOp<float, CPUContext>);

#ifdef WITH_CUDA
REGISTER_CUDA_OPERATOR(TimePlot, TimePlotOp<float, CUDAContext>);
#endif

OPERATOR_SCHEMA(TimePlot)
    .NumInputs(1, 2)
    .NumOutputs(0)
    .ScalarType(TensorProto::FLOAT)
    .SetDoc("Time plot scalar values.")
    .Input(0, "values", "1-D tensor (Tensor<float>)")
    .Input(1, "iter", "1-D tensor (Tensor<float>)")
    .Arg("title", "window title");

SHOULD_NOT_DO_GRADIENT(TimePlot);

}  // namespace caffe2
