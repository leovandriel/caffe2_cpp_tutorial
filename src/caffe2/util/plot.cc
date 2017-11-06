#include "caffe2/util/plot.h"

#include "caffe2/util/window.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <iomanip>
#include <iostream>

namespace caffe2 {

typedef PlotUtil::Series Series;
typedef PlotUtil::Figure Figure;

cv::Scalar color2scalar(const PlotUtil::Color &color) {
  return cv::Scalar(color.b, color.g, color.r);
}

float value2snap(float value) {
  return std::max({pow(10, floor(log10(value))),
                   pow(10, floor(log10(value / 2))) * 2,
                   pow(10, floor(log10(value / 5))) * 5});
}

namespace {
PlotUtil shared_plot;
}

Figure &PlotUtil::Shared(const std::string &window) {
  return shared_plot.Get(window);
}

PlotUtil::Color PlotUtil::Color::Hue(float hue) {
  Color color;
  auto i = (int)hue;
  auto f = (hue - i) * (256 - paleness * 2) + paleness;
  switch (i) {
    case 0:
      color.r = 256 - paleness;
      color.g = f;
      color.b = paleness;
      break;
    case 1:
      color.r = 256 - f;
      color.g = 256 - paleness;
      color.b = paleness;
      break;
    case 2:
      color.r = paleness;
      color.g = 256 - paleness;
      color.b = f;
      break;
    case 3:
      color.r = paleness;
      color.g = 256 - f;
      color.b = 256 - paleness;
      break;
    case 4:
      color.r = f;
      color.g = paleness;
      color.b = 256 - paleness;
      break;
    case 5:
    default:
      color.r = 256 - paleness;
      color.g = paleness;
      color.b = 256 - f;
      break;
  }
  return color;
}

void Series::Bounds(float &x_min, float &x_max, float &y_min, float &y_max,
                    int &n_max, int &p_max) {
  for (auto &d : data_) {
    auto x = d.first, y = d.second;
    if (type_ == Vertical || type_ == Vistogram) {
      x = d.second;
      y = d.first;
    }
    if (type_ != Horizontal) {
      if (x_min > x) {
        x_min = x;
      }
      if (x_max < x) {
        x_max = x;
      }
    }
    if (type_ != Vertical) {
      if (y_min > y) {
        y_min = y;
      }
      if (y_max < y) {
        y_max = y;
      }
    }
  }
  if (n_max < data_.size()) {
    n_max = data_.size();
  }
  if (type_ == Histogram || type_ == Vistogram) {
    p_max = std::max(30, p_max);
  }
}

void Series::Dot(void *b, int x, int y, int r) {
  auto &buffer = *(cv::Mat *)b;
  cv::circle(buffer, {x, y}, r, color2scalar(color_), -1, CV_AA);
}

void Series::Draw(void *b, float x_min, float x_max, float y_min, float y_max,
                  float xs, float xd, float ys, float yd, float x_axis,
                  float y_axis, int unit, float offset) {
  auto &buffer = *(cv::Mat *)b;
  auto color = color2scalar(color_);
  switch (type_) {
    case Line:
    case DotLine:
    case Dots: {
      std::pair<float, float> *last = NULL;
      for (auto &d : data_) {
        cv::Point point((int)(d.first * xs + xd), (int)(d.second * ys + yd));
        if (last) {
          if (type_ == DotLine || type_ == Line) {
            cv::line(
                buffer,
                {(int)(last->first * xs + xd), (int)(last->second * ys + yd)},
                point, color, 1, CV_AA);
          }
        }
        if (type_ == DotLine || type_ == Dots) {
          cv::circle(buffer, point, 2, color, 1, CV_AA);
        }
        last = &d;
      }
    } break;
    case Vistogram:
    case Histogram: {
      auto u = 2 * unit;
      auto o = (int)(2 * u * offset);
      for (auto &d : data_) {
        if (type_ == Histogram) {
          cv::rectangle(
              buffer,
              {(int)(d.first * xs + xd) - u + o, (int)(y_axis * ys + yd)},
              {(int)(d.first * xs + xd) + u + o, (int)(d.second * ys + yd)},
              color, -1, CV_AA);
        } else if (type_ == Vistogram) {
          cv::rectangle(
              buffer,
              {(int)(x_axis * xs + xd), (int)(d.first * ys + yd) - u + o},
              {(int)(d.second * xs + xd), (int)(d.first * ys + yd) + u + o},
              color, -1, CV_AA);
        }
      }

    } break;
    case Horizontal:
    case Vertical: {
      for (auto &d : data_) {
        if (type_ == Horizontal) {
          cv::line(buffer, {(int)(x_min * xs + xd), (int)(d.second * ys + yd)},
                   {(int)(x_max * xs + xd), (int)(d.second * ys + yd)}, color,
                   1, CV_AA);
        } else if (type_ == Vertical) {
          cv::line(buffer, {(int)(d.second * xs + xd), (int)(y_min * ys + yd)},
                   {(int)(d.second * xs + xd), (int)(y_max * ys + yd)}, color,
                   1, CV_AA);
        }
      }
    } break;
  }
}

void Figure::Draw(void *b, float x_min, float x_max, float y_min, float y_max,
                  int n_max, int p_max) {
  auto &buffer = *(cv::Mat *)b;

  // draw background and sub axis square
  cv::rectangle(buffer, {0, 0}, {buffer.cols, buffer.rows},
                color2scalar(background_color_), -1, CV_AA);
  cv::rectangle(buffer, {border_size_, border_size_},
                {buffer.cols - border_size_, buffer.rows - border_size_},
                color2scalar(sub_axis_color_), 1, CV_AA);

  // size of the plotting area
  auto w_plot = buffer.cols - 2 * border_size_;
  auto h_plot = buffer.rows - 2 * border_size_;

  // add padding inside graph (histograms get extra)
  if (p_max) {
    auto dx = p_max * (x_max - x_min) / w_plot;
    auto dy = p_max * (y_max - y_min) / h_plot;
    x_min -= dx;
    x_max += dx;
    y_min -= dy;
    y_max += dy;
  }

  // adjust value range if aspect ratio square
  if (aspect_square_) {
    if (h_plot * (x_max - x_min) < w_plot * (y_max - y_min)) {
      auto dx = w_plot * (y_max - y_min) / h_plot - (x_max - x_min);
      x_min -= dx / 2;
      x_max += dx / 2;
    } else if (w_plot * (y_max - y_min) < h_plot * (x_max - x_min)) {
      auto dy = h_plot * (x_max - x_min) / w_plot - (y_max - y_min);
      y_min -= dy / 2;
      y_max += dy / 2;
    }
  }

  // calc where to draw axis
  auto x_axis = std::max(x_min, std::min(x_max, 0.f));
  auto y_axis = std::max(y_min, std::min(y_max, 0.f));

  // calc sub axis grid size
  auto x_grid =
      (x_max != x_min ? value2snap((x_max - x_min) / floor(w_plot / grid_size_))
                      : 1);
  auto y_grid =
      (y_max != x_min ? value2snap((y_max - y_min) / floor(h_plot / grid_size_))
                      : 1);

  // calc affine transform value space to plot space
  auto xs = (x_max != x_min ? (buffer.cols - 2 * border_size_) / (x_max - x_min)
                            : 1.f);
  auto xd = border_size_ - x_min * xs;
  auto ys = (y_max != y_min ? (buffer.rows - 2 * border_size_) / (y_min - y_max)
                            : 1.f);
  auto yd = buffer.rows - y_min * ys - border_size_;

  // safe unit for showing points
  auto unit =
      std::max(1, ((int)std::min(buffer.cols, buffer.rows) - 2 * border_size_) /
                      n_max / 10);

  // draw sub axis
  for (auto x = ceil(x_min / x_grid) * x_grid; x <= x_max; x += x_grid) {
    cv::line(buffer, {(int)(x * xs + xd), border_size_},
             {(int)(x * xs + xd), buffer.rows - border_size_},
             color2scalar(sub_axis_color_), 1, CV_AA);
    std::ostringstream out;
    out << std::setprecision(4) << (x == 0 ? 0 : x);
    int baseline;
    cv::Size size =
        getTextSize(out.str(), cv::FONT_HERSHEY_SIMPLEX, 0.3, 1.0, &baseline);
    cv::Point org(x * xs + xd - size.width / 2,
                  buffer.rows - border_size_ + 5 + size.height);
    cv::putText(buffer, out.str().c_str(), org, cv::FONT_HERSHEY_SIMPLEX, 0.3,
                color2scalar(text_color_), 1.0);
  }
  for (auto y = ceil(y_min / y_grid) * y_grid; y <= y_max; y += y_grid) {
    cv::line(buffer, {border_size_, (int)(y * ys + yd)},
             {buffer.cols - border_size_, (int)(y * ys + yd)},
             color2scalar(sub_axis_color_), 1, CV_AA);
    std::ostringstream out;
    out << std::setprecision(4) << (y == 0 ? 0 : y);
    int baseline;
    cv::Size size =
        getTextSize(out.str(), cv::FONT_HERSHEY_SIMPLEX, 0.3, 1.0, &baseline);
    cv::Point org(border_size_ - 5 - size.width, y * ys + yd + size.height / 2);
    cv::putText(buffer, out.str().c_str(), org, cv::FONT_HERSHEY_SIMPLEX, 0.3,
                color2scalar(text_color_), 1.0);
  }

  // draw axis
  cv::line(buffer, {(int)(x_axis * xs + xd), border_size_},
           {(int)(x_axis * xs + xd), buffer.rows - border_size_},
           color2scalar(axis_color_), 1, CV_AA);
  cv::line(buffer, {border_size_, (int)(y_axis * ys + yd)},
           {buffer.cols - border_size_, (int)(y_axis * ys + yd)},
           color2scalar(text_color_), 1, CV_AA);

  // draw plot
  auto index = 0;
  for (auto &s : series_) {
    if (s.Collides()) {
      index++;
    }
  }
  std::max((int)series_.size() - 1, 1);
  for (auto s = series_.rbegin(); s != series_.rend(); ++s) {
    if (s->Collides()) {
      index--;
    }
    s->Draw(&buffer, x_min, x_max, y_min, y_max, xs, xd, ys, yd, x_axis, y_axis,
            unit, (float)index / series_.size());
  }

  // draw label names
  index = 0;
  for (auto &s : series_) {
    auto name = s.Label();
    int baseline;
    cv::Size size =
        getTextSize(name, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1.0, &baseline);
    cv::Point org(buffer.cols - border_size_ - size.width - 17,
                  border_size_ + 15 * index + 15);
    auto shadow = true;
    cv::putText(buffer, name.c_str(),
                {org.x + (shadow ? 1 : 0), org.y + (shadow ? 1 : 0)},
                cv::FONT_HERSHEY_SIMPLEX, 0.4, color2scalar(background_color_),
                (shadow ? 1.0 : 2.0));
    cv::putText(buffer, name.c_str(), org, cv::FONT_HERSHEY_SIMPLEX, 0.4,
                color2scalar(text_color_), 1.0);
    cv::circle(buffer, {buffer.cols - border_size_ - 10 + 1, org.y - 3 + 1}, 3,
               color2scalar(background_color_), -1, CV_AA);
    s.Dot(&buffer, buffer.cols - border_size_ - 10, org.y - 3, 3);
    index++;
  }
}

void Figure::Show(bool flush) {
  auto x_min = (include_zero_x_ ? 0.f : FLT_MAX);
  auto x_max = (include_zero_x_ ? 0.f : FLT_MIN);
  auto y_min = (include_zero_y_ ? 0.f : FLT_MAX);
  auto y_max = (include_zero_y_ ? 0.f : FLT_MIN);
  auto n_max = 0;
  auto p_max = grid_padding_;

  // find value bounds
  for (auto &s : series_) {
    s.Bounds(x_min, x_max, y_min, y_max, n_max, p_max);
  }

  if (n_max) {
    cv::Rect rect;
    auto buffer = caffe2::getBuffer(window_.c_str(), rect)(rect);
    Draw(&buffer, x_min, x_max, y_min, y_max, n_max, p_max);
    caffe2::showBuffer(window_.c_str(), flush);
    // cvWaitKey(1);
  }
}

}  // namespace caffe2
