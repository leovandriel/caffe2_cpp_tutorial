#include "caffe2/util/tensor.h"

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

namespace caffe2 {

const auto screen_width = 1600;
const auto window_padding = 4;

cv::Mat TensorUtil::toImage(int index, float mean) {
  auto count = tensor_.dim(0), depth = tensor_.dim(1), height = tensor_.dim(2), width = tensor_.dim(3);
  CHECK(index < count);
  auto data = tensor_.data<float>() + (index * width * height);
  vector<cv::Mat> channels(depth);
  for (auto& j: channels) {
    j = cv::Mat(height, width, CV_32F, (void*)data);
    data += (width * height);
  }
  cv::Mat image;
  cv::merge(channels, image);
  image.convertTo(image, CV_8UC3, 1.0, mean);
  return image;
}

void TensorUtil::showImage(int width, int height, int index, const std::string& name, int offset, float mean) {
  auto title = name + "-" + std::to_string(index);
  auto image = toImage(index, mean);
  cv::resize(image, image, cv::Size(width, height));
  cv::namedWindow(title, cv::WINDOW_AUTOSIZE);
  auto max_cols = screen_width / (image.cols + window_padding);
  cv::moveWindow(title, (offset % max_cols) * (image.cols + window_padding), (offset / max_cols) * (image.rows + window_padding));
  cv::imshow(title, image);
  cv::waitKey(1);
}

void TensorUtil::showImages(int width, int height, const std::string& name, float mean) {
  for (auto i = 0; i < tensor_.dim(0); i++) {
    auto title = name + "-" + std::to_string(i);
    auto image = toImage(i, mean);
    cv::resize(image, image, cv::Size(width, height));
    cv::namedWindow(title, cv::WINDOW_AUTOSIZE);
    auto max_cols = screen_width / (image.cols + window_padding);
    cv::moveWindow(title, (i % max_cols) * (image.cols + window_padding), (i / max_cols) * (image.rows + window_padding));
    cv::imshow(title, image);
    cv::waitKey(1);
  }
}

void TensorUtil::writeImages(const std::string& name, float mean) {
  auto count = tensor_.dim(0);
  for (int i = 0; i < count; i++) {
    auto image = toImage(i, mean);
    auto filename = name + "_" + std::to_string(i) + ".jpg";
    vector<int> params({ CV_IMWRITE_JPEG_QUALITY, 90 });
    CHECK(cv::imwrite(filename, image, params));
    // vector<uchar> buffer;
    // cv::imencode(".jpg", image, buffer, params);
    // std::ofstream image_file(filename, std::ios::out | std::ios::binary);
    // if (image_file.is_open()) {
    //   image_file.write((char*)&buffer[0], buffer.size());
    //   image_file.close();
    // }
  }
}

}  // namespace caffe2
