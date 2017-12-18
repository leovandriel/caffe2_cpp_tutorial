#include "caffe2/operator/time_plot_op.h"

#include "caffe2/util/plot.h"
#include "caffe2/util/tensor.h"

#ifdef WITH_CUDA
#include <caffe2/core/common_cudnn.h>
#include <caffe2/core/context_gpu.h>
#include <caffe2/core/types.h>
#endif

namespace caffe2 {

void time_plot(const TensorCPU &X, const std::string &label_,
               const std::string &window_, const int &step_, const int &index_,
               int &step_count_, float &index_sum_, float &value_sum_,
               float &sq_sum_) {
  step_count_++;
  value_sum_ += X.data<float>()[0];
  sq_sum_ += X.data<float>()[0] * X.data<float>()[0];
  index_sum_ += index_;
  if (step_count_ >= step_) {
    auto &figure = PlotUtil::Shared(window_);
    auto index = index_sum_ / step_count_, mean = value_sum_ / step_count_,
         stdev = sqrtf(sq_sum_ / step_count_ - mean * mean);
    value_sum_ = 0.f;
    sq_sum_ = 0.f;
    index_sum_ = 0.f;
    step_count_ = 0;
    auto color = figure.Get(label_).Add(index, mean).Color();
    figure.Get(label_ + "_range")
        .Add(index, {mean - stdev, mean + stdev})
        .Type(PlotUtil::Range)
        .Color(color.Alpha(64))
        .Legend(false);
    figure.Show(true);
  }
}

template <>
bool TimePlotOp<float, CPUContext>::RunOnDevice() {
  auto index = index_++;
  if (InputSize() > 1) {
    index = TensorCPU(Input(1)).data<int64_t>()[0];
  }
  time_plot(Input(0), label_, window_, step_, index, step_count_, index_sum_,
            value_sum_, sq_sum_);
  return true;
}

#ifdef WITH_CUDA
template <>
bool TimePlotOp<float, CUDAContext>::RunOnDevice() {
  auto index = index_++;
  if (InputSize() > 1) {
    index = InputBlob(1).Get<TensorCPU>().data<int64_t>()[0];
  }
  time_plot(TensorCPU(Input(0)), label_, window_, step_, index, step_count_,
            index_sum_, value_sum_, sq_sum_),
      sq_sum_;
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
