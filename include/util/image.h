#ifndef IMAGE_H
#define IMAGE_H

#include "caffe2/core/net.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

namespace caffe2 {

TensorCPU readImageTensor(const string &filename, int size, float mean) {
  // load image
  auto image = cv::imread(filename); // CV_8UC3
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

  // convert to float, normalize to mean 128
  image.convertTo(image, CV_32FC3, 1.0, mean);
  // std::cout << "value range: (" << *std::min_element((float *)image.datastart, (float *)image.dataend) << ", " << *std::max_element((float *)image.datastart, (float *)image.dataend) << ")" << std::endl;

  // convert NHWC to NCHW
  vector<cv::Mat> channels(3);
  cv::split(image, channels);
  std::vector<float> data;
  for (auto &c: channels) {
    data.insert(data.end(), (float *)c.datastart, (float *)c.dataend);
  }

  // create tensor
  std::vector<TIndex> dims({1, image.channels(), image.rows, image.cols});
  TensorCPU tensor(dims, data, NULL);

  return tensor;
}

}  // namespace caffe2

#endif  // IMAGE_H
