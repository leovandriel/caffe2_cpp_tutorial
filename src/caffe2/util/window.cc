#include "caffe2/util/window.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace caffe2 {

namespace {
WindowUtil shared_window;
}

void WindowUtil::ResizeWindow(cv::Rect rect) {
  PositionWindow({rect.x, rect.y});
  SizeWindow({rect.width, rect.height});
}

void WindowUtil::SizeWindow(cv::Size size) {
  auto buffer = cv::Mat(size, CV_8UC3, 0.0);
  if (buffer_.cols > 0 && buffer_.rows > 0 && size.width > 0 &&
      size.height > 0) {
    cv::Rect inter(0, 0, std::min(buffer_.cols, size.width),
                   std::min(buffer_.rows, size.height));
    buffer_(inter).copyTo(buffer(inter));
  }
  buffer_ = buffer;
  Show();
}

void WindowUtil::PositionWindow(cv::Point position) {
  position_ = position;
  cv::moveWindow(title_, position.x, position.y);
}

void WindowUtil::EnsureWindow(cv::Rect rect) {
  if (rect.x + rect.width > buffer_.cols ||
      rect.y + rect.height > buffer_.rows) {
    SizeWindow({std::max(buffer_.cols, rect.x + rect.width),
                std::max(buffer_.rows, rect.y + rect.height)});
  }
}

void WindowUtil::Show() {
  if (buffer_.cols > 0 && buffer_.rows > 0) {
    cv::imshow(title_.c_str(), buffer_);
    cvWaitKey(1);
  }
}

void WindowUtil::ResizeView(const std::string &name, cv::Rect rect) {
  views_[name].rect = rect;
}

void WindowUtil::SizeView(const std::string &name, cv::Size size) {
  auto rect = views_[name].rect;
  rect.width = size.width;
  rect.height = size.height;
  views_[name].rect = rect;
}

void WindowUtil::OffsetView(const std::string &name, cv::Point offset) {
  auto rect = views_[name].rect;
  rect.x = offset.x;
  rect.y = offset.y;
  views_[name].rect = rect;
}

void WindowUtil::AutosizeView(const std::string &name) {
  auto rect = views_[name].rect;
  rect.width = 0;
  rect.height = 0;
  views_[name].rect = rect;
}

void WindowUtil::TitleView(const std::string &name, const std::string &title) {
  views_[name].title = title;
}

void WindowUtil::ShowText(const std::string &name, const std::string &text,
                          cv::Point position) {
  auto rect = views_[name].rect;
  auto face = cv::FONT_HERSHEY_SIMPLEX;
  auto scale = 0.5;
  auto thickness = 1.0;
  int baseline;
  cv::Size size = getTextSize(text, face, scale, thickness, &baseline);
  cv::Point org(rect.x + position.x, rect.y + size.height + position.y);
  cv::rectangle(buffer_, org + cv::Point(-1, 2),
                org + cv::Point(size.width + 1, -size.height - 1),
                cv::Scalar::all(224), -1);
  cv::putText(buffer_, text.c_str(), org, face, scale, cv::Scalar::all(32),
              thickness);
}

void WindowUtil::ShowFrame(const std::string &name, const std::string &title,
                           cv::Scalar foreground, cv::Scalar background,
                           cv::Scalar text) {
  auto rect = views_[name].rect;
  cv::rectangle(buffer_, {rect.x, rect.y},
                {rect.x + rect.width - 1, rect.y + rect.height - 1}, background,
                1);
  cv::rectangle(buffer_, {rect.x + 1, rect.y + 1},
                {rect.x + rect.width - 2, rect.y + rect.height - 2}, foreground,
                1);
  cv::rectangle(buffer_, {rect.x + 2, rect.y + 2},
                {rect.x + rect.width - 3, rect.y + 16}, foreground, -1);
  int baseline;
  cv::Size size =
      getTextSize(title.c_str(), cv::FONT_HERSHEY_PLAIN, 1.0, 1.0, &baseline);
  cv::putText(buffer_, title.c_str(),
              {rect.x + 2 + (rect.width - size.width) / 2, rect.y + 14},
              cv::FONT_HERSHEY_PLAIN, 1.0, text, 1.0);
}

void WindowUtil::ShowImage(const std::string &name, const cv::Mat &image,
                           bool flush) {
  auto &view = views_[name];
  auto &rect = view.rect;
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
  ShowBuffer(name, flush);
}

void WindowUtil::ClearView(const std::string &name, bool flush,
                           cv::Scalar background) {
  auto &rect = views_[name].rect;
  cv::rectangle(buffer_, {rect.x, rect.y},
                {rect.x + rect.width - 1, rect.y + rect.height - 1}, background,
                -1);
  ShowBuffer(name, flush);
}

cv::Mat WindowUtil::GetBuffer(const std::string &name, cv::Rect &rect) {
  auto &view = views_[name];
  EnsureWindow(view.rect);
  rect = view.rect;
  return buffer_;
}

void WindowUtil::ShowBuffer(const std::string &name, bool flush) {
  auto &view = views_[name];
  if (!view.frameless) {
    ShowFrame(name, view.title.size() ? view.title : name);
  }
  if (flush) {
    Show();
  }
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

void setWindowTitle(const char *name, const char *title) {
  shared_window.TitleView(name, title);
}

void imshow(const char *name, const cv::Mat &mat, bool flush) {
  shared_window.ShowImage(name, mat, flush);
}

cv::Mat getBuffer(const char *name, cv::Rect &rect) {
  return shared_window.GetBuffer(name, rect);
}

void showBuffer(const char *name, bool flush) {
  shared_window.ShowBuffer(name, flush);
}

void clearWindow(const char *name, bool flush) {
  shared_window.ClearView(name, flush);
}

}  // namespace caffe2
