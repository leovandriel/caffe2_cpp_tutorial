#ifndef IMAGE_H
#define IMAGE_H

#include "caffe2/core/net.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

namespace caffe2 {

template <typename T>
TensorCPU imageToTensor(const cv::Mat image) {
  // std::cout << "value range: (" << *std::min_element((T *)image.datastart, (T *)image.dataend) << ", " << *std::max_element((T *)image.datastart, (T *)image.dataend) << ")" << std::endl;

  // convert NHWC to NCHW
  vector<cv::Mat> channels(3);
  cv::split(image, channels);
  std::vector<T> data;
  for (auto &c: channels) {
    data.insert(data.end(), (T *)c.datastart, (T *)c.dataend);
  }

  // create tensor
  std::vector<TIndex> dims({ 1, image.channels(), image.rows, image.cols });
  return TensorCPU(dims, data, NULL);
}

TensorCPU readImageTensor(const string &filename, int size, float mean = 128, TensorProto::DataType data_type = TensorProto_DataType_FLOAT) {
  // load image
  auto image = cv::imread(filename); // CV_8UC3 uchar
  // std::cout << "image size: " << image.size() << std::endl;

  if (!image.cols || !image.rows) {
    return TensorCPU();
  }

  // scale image to fit
  cv::Size scale(std::max(size * image.cols / image.rows, size), std::max(size, size * image.rows / image.cols));
  cv::resize(image, image, scale);
  // std::cout << "scaled size: " << image.size() << std::endl;

  // crop image to fit
  cv::Rect crop((image.cols - size) / 2, (image.rows - size) / 2, size, size);
  image = image(crop);
  // std::cout << "cropped size: " << image.size() << std::endl;

  std::vector<float> data;
  switch (data_type) {
  case TensorProto_DataType_FLOAT:
    image.convertTo(image, CV_32FC3, 1.0, -mean);
    return imageToTensor<float>(image);
  case TensorProto_DataType_INT8:
    image.convertTo(image, CV_8SC3, 1.0, -mean);
    return imageToTensor<int8_t>(image);
  case TensorProto_DataType_UINT8:
    return imageToTensor<uint8_t>(image);
  default:
    LOG(FATAL) << "datatype " << data_type << " not implemented";
    // convert to float, normalize to mean 128
  }
}

}  // namespace caffe2

#endif  // IMAGE_H
