#ifndef IMAGE_H
#define IMAGE_H

#include "caffe2/core/net.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

namespace caffe2 {

template <typename T>
TensorCPU readImageTensorImp(const std::vector<std::string> &filenames, int size, std::vector<int> &indices, float mean, TensorProto::DataType type) {
  std::vector<T> data;
  data.reserve(filenames.size() * 3 * size * size);
  auto count = 0;

  for (auto &filename: filenames) {
    // load image
    auto image = cv::imread(filename); // CV_8UC3 uchar
    // std::cout << "image size: " << image.size() << std::endl;

    if (!image.cols || !image.rows) {
      count++;
      continue;
    }

    // scale image to fit
    cv::Size scale(std::max(size * image.cols / image.rows, size), std::max(size, size * image.rows / image.cols));
    cv::resize(image, image, scale);
    // std::cout << "scaled size: " << image.size() << std::endl;

    // crop image to fit
    cv::Rect crop((image.cols - size) / 2, (image.rows - size) / 2, size, size);
    image = image(crop);
    // std::cout << "cropped size: " << image.size() << std::endl;

    switch (type) {
    case TensorProto_DataType_FLOAT:
      image.convertTo(image, CV_32FC3, 1.0, -mean);
      break;
    case TensorProto_DataType_INT8:
      image.convertTo(image, CV_8SC3, 1.0, -mean);
      break;
    default:
      break;
    }
    // std::cout << "value range: (" << *std::min_element((T *)image.datastart, (T *)image.dataend) << ", " << *std::max_element((T *)image.datastart, (T *)image.dataend) << ")" << std::endl;

    CHECK(image.channels() == 3);
    CHECK(image.rows == size);
    CHECK(image.cols == size);

    // convert NHWC to NCHW
    vector<cv::Mat> channels(3);
    cv::split(image, channels);
    for (auto &c: channels) {
      data.insert(data.end(), (T *)c.datastart, (T *)c.dataend);
    }

    indices.push_back(count++);
  }

  // create tensor
  std::vector<TIndex> dims({ (TIndex)indices.size(), 3, size, size });
  return TensorCPU(dims, data, NULL);
}

TensorCPU readImageTensor(const std::vector<std::string> &filenames, int size, std::vector<int> &indices, float mean = 128, TensorProto::DataType type = TensorProto_DataType_FLOAT) {
    switch (type) {
    case TensorProto_DataType_FLOAT:
      return readImageTensorImp<float>(filenames, size, indices, mean, type);
    case TensorProto_DataType_INT8:
      return readImageTensorImp<int8_t>(filenames, size, indices, mean, type);
    case TensorProto_DataType_UINT8:
      return readImageTensorImp<uint8_t>(filenames, size, indices, mean, type);
    default:
      LOG(FATAL) << "datatype " << type << " not implemented";
    }
}

TensorCPU readImageTensor(const std::string &filename, int size) {
  std::vector<int> indices;
  return readImageTensor({ filename }, size, indices);
}

TensorCPU imageToTensor(cv::Mat &image, float mean = 128) {
  std::vector<float> data;
  image.convertTo(image, CV_32FC3, 1.0, -mean);
  vector<cv::Mat> channels(3);
  cv::split(image, channels);
  for (auto &c: channels) {
    data.insert(data.end(), (float *)c.datastart, (float *)c.dataend);
  }
  std::vector<TIndex> dims({ 1, 3, image.rows, image.cols });
  return TensorCPU(dims, data, NULL);
}

TensorCPU scaleImageTensor(const TensorCPU &tensor, int width, int height) {
  auto count = tensor.dim(0), dim_c = tensor.dim(1), dim_h = tensor.dim(2), dim_w = tensor.dim(3);
  std::vector<float> output;
  output.reserve(count * dim_c * height * width);
  auto input = tensor.data<float>();
  vector<cv::Mat> channels(dim_c);
  for (int i = 0; i < count; i++) {
    for (auto &j: channels) {
      j = cv::Mat(dim_h, dim_w, CV_32F, (void *)input);
      input += (dim_w * dim_h);
    }
    cv::Mat image;
    cv::merge(channels, image);
    // image.convertTo(image, CV_8UC3, 1.0, mean);

    cv::resize(image, image, cv::Size(width, height));

    // image.convertTo(image, CV_32FC3, 1.0, -mean);
    cv::split(image, channels);
    for (auto &c: channels) {
      output.insert(output.end(), (float *)c.datastart, (float *)c.dataend);
    }
  }
  std::vector<TIndex> dims({ count, dim_c, height, width });
  return TensorCPU(dims, output, NULL);
}

}  // namespace caffe2

#endif  // IMAGE_H
