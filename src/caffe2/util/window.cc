#include "caffe2/util/window.h"

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

namespace caffe2 {

namespace {
WindowUtil shared_window("main");
}

void WindowUtil::ResizeWindow(cv::Rect rect) {
  PositionWindow({rect.x, rect.y});
  SizeWindow({rect.width, rect.height});
}

void WindowUtil::SizeWindow(cv::Size size) {
  buffer_ = cv::Mat(size, CV_8UC3);
  // TODO: copy old content?
  Show();
}

void WindowUtil::PositionWindow(cv::Point position) {
  position_ = position;
  cv::moveWindow(title_, position.x, position.y);
}

void WindowUtil::EnsureWindow(cv::Rect rect) {
  if (rect.x + rect.width > buffer_.cols ||
      rect.y + rect.height > buffer_.rows) {
    SizeWindow({rect.x + rect.width, rect.y + rect.height});
  }
}

void WindowUtil::Show() {
  if (buffer_.cols > 0 && buffer_.rows > 0) {
    cv::imshow(title_.c_str(), buffer_);
    cvWaitKey(1);
  }
}

void WindowUtil::ResizeView(const std::string &name, cv::Rect rect) {
  rects_[name] = rect;
}

void WindowUtil::SizeView(const std::string &name, cv::Size size) {
  auto rect = rects_[name];
  rect.width = size.width;
  rect.height = size.height;
  rects_[name] = rect;
}

void WindowUtil::OffsetView(const std::string &name, cv::Point offset) {
  auto rect = rects_[name];
  rect.x = offset.x;
  rect.y = offset.y;
  rects_[name] = rect;
}

void WindowUtil::AutosizeView(const std::string &name) {
  auto rect = rects_[name];
  rect.width = 0;
  rect.height = 0;
  rects_[name] = rect;
}

void WindowUtil::ShowImage(const std::string &name, const cv::Mat &image) {
  auto rect = rects_[name];
  if (rect.width == 0 && rect.height == 0) {
    rect.width = image.cols;
    rect.height = image.rows;
  }
  EnsureWindow(rect);
  if (image.cols != rect.width || image.rows != rect.height) {
    cv::Mat resized;
    resize(image, resized, {rect.width, rect.height});
    resized.copyTo(buffer_(rect));
  } else {
    image.copyTo(buffer_(rect));
  }
  Show();
}

void WindowUtil::SetTitle(const std::string &title) {
  title_ = title;
  cv::namedWindow(title, cv::WINDOW_AUTOSIZE);
}

void superWindow(const char *title, int width, int height) {
  shared_window.SetTitle(title);
  shared_window.SizeWindow({width, height});
}

void moveWindow(const char *name, int x, int y) {
  shared_window.OffsetView(name, {x, y});
}

void resizeWindow(const char *name, int width, int height) {
  shared_window.SizeView(name, {width, height});
}

void autosizeWindow(const char *name) { shared_window.AutosizeView(name); }

void imshow(const char *name, const cv::Mat &mat) {
  shared_window.ShowImage(name, mat);
}

}  // namespace caffe2
