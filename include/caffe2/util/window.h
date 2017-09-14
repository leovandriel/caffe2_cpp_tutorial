#ifndef UTIL_WINDOW_H
#define UTIL_WINDOW_H

#include "opencv2/opencv.hpp"

#include <map>

namespace caffe2 {

class WindowUtil {
 public:
  WindowUtil(const std::string &title, cv::Size size = {0, 0}) {
    SetTitle(title);
    SizeWindow(size);
  }

  void ResizeWindow(cv::Rect rect);
  void SizeWindow(cv::Size size);
  void PositionWindow(cv::Point position);
  void EnsureWindow(cv::Rect rect);

  void ResizeView(const std::string &name, cv::Rect rect);
  void SizeView(const std::string &name, cv::Size size);
  void OffsetView(const std::string &name, cv::Point offset);
  void AutosizeView(const std::string &name);

  void ShowImage(const std::string &name, const cv::Mat &image);
  void SetTitle(const std::string &title);
  void Show();

 protected:
  cv::Point position_;
  cv::Mat buffer_;
  std::string title_;
  std::map<std::string, cv::Rect> rects_;
};

void superWindow(const char *title, int width = 0, int height = 0);
void moveWindow(const char *name, int x, int y);
void resizeWindow(const char *name, int width, int height);
void autosizeWindow(const char *name);
void imshow(const char *name, const cv::Mat &mat);

}  // namespace caffe2

#endif  // UTIL_WINDOW_H
