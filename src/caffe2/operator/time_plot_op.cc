#include "caffe2/operator/time_plot_op.h"

#include "caffe2/util/tensor.h"

#ifdef WITH_CUDA
#include "caffe2/core/common_cudnn.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/core/types.h"
#endif

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "cvplot/cvplot.h"

namespace caffe2 {

void time_plot(const TensorCPU &values, std::vector<float> &past,
               const std::string &name, const std::string &label) {
  const auto *data = values.data<float>();
  for (auto d = data, e = data + values.size(); d != e; d++) {
    past.push_back(*d);
  }
  CvPlot::plot(name, &past[0], past.size(), 1);
  CvPlot::label(label);
}

template <>
bool TimePlotOp<float, CPUContext>::RunOnDevice() {
  CvPlot::clear(name_);
  for (auto i = 0; i < labels_.size(); ++i) {
    time_plot(Input(i), pasts_[i], name_, labels_[i]);
  }
  return true;
}

#ifdef WITH_CUDA
template <>
bool TimePlotOp<float, CUDAContext>::RunOnDevice() {
  CvPlot::clear(name_);
  for (auto i = 0; i < labels_.size(); ++i) {
    time_plot(TensorCPU(Input(i)), pasts_[i], name_, labels_[i]);
  }
  return true;
}
#endif

REGISTER_CPU_OPERATOR(TimePlot, TimePlotOp<float, CPUContext>);

#ifdef WITH_CUDA
REGISTER_CUDA_OPERATOR(TimePlot, TimePlotOp<float, CUDAContext>);
#endif

OPERATOR_SCHEMA(TimePlot)
    .NumInputs(1, 10)
    .NumOutputs(0)
    .ScalarType(TensorProto::FLOAT)
    .SetDoc("Time plot scalar values.")
    .Input(0, "values", "1-D tensor (Tensor<float>)")
    .Arg("title", "window title");

SHOULD_NOT_DO_GRADIENT(TimePlot);

}  // namespace caffe2
