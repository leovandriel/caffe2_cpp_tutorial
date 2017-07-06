#ifndef MATH_H
#define MATH_H

#include "caffe2/core/tensor.h"

namespace caffe2 {

template<typename C>
void mean_stdev_tensor(const Tensor<C> &tensor, float *mean_out, float *stdev_out) {
  auto data = tensor.template data<float>();
  std::vector<float> values(data, data + tensor.size());
  float sum = std::accumulate(values.begin(), values.end(), 0.0);
  float mean = sum / values.size();
  std::vector<float> diff(values.size());
  std::transform(values.begin(), values.end(), diff.begin(), [mean](float x) { return x - mean; });
  float sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
  float stdev = std::sqrt(sq_sum / values.size());
  if (mean_out) *mean_out = mean;
  if (stdev_out) *stdev_out = stdev;
}

template<typename C>
void normalize_tensor(Tensor<C> &tensor, float scale) {
  float mean = 0, stdev = 0;
  mean_stdev_tensor(tensor, &mean, &stdev);
  float s = scale / std::max(stdev, 1e-4f);
  affine_transform_tensor(tensor, s, -mean * s);
}

template<typename C>
void affine_transform_tensor(Tensor<C> &tensor, float scale, float bias = 0) {
  auto data = tensor.template mutable_data<float>();
  for (auto b = data, e = b + tensor.size(); b != e; b++) {
    *b = *b * scale + bias;
  }
}

template<typename C>
void add_tensor(Tensor<C> &tensor, const Tensor<C> &tensor_add) {
  CHECK(tensor.size() == tensor_add.size());
  auto data = tensor.template mutable_data<float>();
  auto b_add = tensor_add.template data<float>();
  for (auto b = data, e = b + tensor.size(); b != e; b++, b_add++) {
    *b += *b_add;
  }
}

}  // namespace caffe2

#endif  // MATH_H
