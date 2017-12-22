#include "cvplot/plot.h"

#include "cvplot/window.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iomanip>
#include <iostream>

#define EXPECT_EQ(a__, b__)                                                    \
  do {                                                                         \
    if ((a__) != (b__)) {                                                      \
      std::cerr << "Incorrect " << #a__ << " (" << (a__) << "), should equal " \
                << (b__) << std::endl;                                         \
      exit(-1);                                                                \
    }                                                                          \
  } while (0)

#define EXPECT_DIMS_DEPTH(dims__, depth__) \
  EXPECT_EQ(dims_, dims__);                \
  EXPECT_EQ(depth_, depth__);

namespace cvplot {

cv::Scalar color2scalar(const Color &color) {
  return cv::Scalar(color.b, color.g, color.r);
}

float value2snap(float value) {
  return std::max({pow(10, floor(log10(value))),
                   pow(10, floor(log10(value / 2))) * 2,
                   pow(10, floor(log10(value / 5))) * 5});
}

namespace {
Plot shared_plot;
}

Plot::Figure &Plot::shared(const std::string &window) {
  return shared_plot.figure(window);
}

Plot::Series &Plot::Series::dims(int dims) {
  if (dims_ != dims) {
    EXPECT_EQ(dims_, 0);
    dims_ = dims;
  }
  return *this;
}

Plot::Series &Plot::Series::depth(int depth) {
  if (depth_ != depth) {
    EXPECT_EQ(depth_, 0);
    depth_ = depth;
  }
  return *this;
}

void Plot::Series::ensure(int dims, int depth) {
  if (dims_ == 0) {
    dims_ = dims;
  }
  if (depth_ == 0) {
    depth_ = depth;
  }
  EXPECT_DIMS_DEPTH(dims, depth);
}

Plot::Series &Plot::Series::clear() {
  entries_.clear();
  data_.clear();
  return *this;
}

Plot::Series &Plot::Series::type(enum Type type) {
  type_ = type;
  return *this;
}

Plot::Series &Plot::Series::color(Color color) {
  color_ = color;
  return *this;
}

Plot::Series &Plot::Series::legend(bool legend) {
  legend_ = legend;
  return *this;
}

Plot::Series &Plot::Series::add(
    const std::vector<std::pair<float, float>> &data) {
  ensure(1, 1);
  for (const auto &d : data) {
    entries_.push_back(data_.size());
    data_.push_back(d.first);
    data_.push_back(d.second);
  }
  return *this;
}

Plot::Series &Plot::Series::add(
    const std::vector<std::pair<float, Point2>> &data) {
  ensure(1, 2);
  for (const auto &d : data) {
    entries_.push_back(data_.size());
    data_.push_back(d.first);
    data_.push_back(d.second.x);
    data_.push_back(d.second.y);
  }
  return *this;
}

Plot::Series &Plot::Series::addValue(const std::vector<float> &values) {
  std::vector<std::pair<float, float>> data(values.size());
  auto i = 0;
  for (auto &d : data) {
    d.first = i + entries_.size();
    d.second = values[i++];
  }
  return add(data);
}

Plot::Series &Plot::Series::addValue(const std::vector<Point2> &values) {
  std::vector<std::pair<float, Point2>> data(values.size());
  auto i = 0;
  for (auto &d : data) {
    d.first = i + entries_.size();
    d.second = values[i++];
  }
  return add(data);
}

Plot::Series &Plot::Series::add(float key, float value) {
  return add(std::vector<std::pair<float, float>>({{key, value}}));
}

Plot::Series &Plot::Series::add(float key, Point2 value) {
  return add(std::vector<std::pair<float, Point2>>({{key, value}}));
}

Plot::Series &Plot::Series::addValue(float value) {
  return addValue(std::vector<float>({value}));
}

Plot::Series &Plot::Series::addValue(float value_a, float value_b) {
  return addValue(std::vector<Point2>({{value_a, value_b}}));
}

Plot::Series &Plot::Series::set(
    const std::vector<std::pair<float, float>> &data) {
  dims(1).depth(1);
  clear();
  return add(data);
}

Plot::Series &Plot::Series::set(
    const std::vector<std::pair<float, Point2>> &data) {
  dims(1).depth(2);
  clear();
  return add(data);
}

Plot::Series &Plot::Series::setValue(const std::vector<float> &values) {
  std::vector<std::pair<float, float>> data(values.size());
  auto i = 0;
  for (auto &d : data) {
    d.first = i;
    d.second = values[i++];
  }
  return set(data);
}

Plot::Series &Plot::Series::setValue(const std::vector<Point2> &values) {
  std::vector<std::pair<float, Point2>> data(values.size());
  auto i = 0;
  for (auto &d : data) {
    d.first = i;
    d.second = values[i++];
  }
  return set(data);
}

Plot::Series &Plot::Series::set(float key, float value) {
  return set(std::vector<std::pair<float, float>>({{key, value}}));
}

Plot::Series &Plot::Series::set(float key, float value_a, float value_b) {
  return set(
      std::vector<std::pair<float, Point2>>({{key, {value_a, value_b}}}));
}

Plot::Series &Plot::Series::setValue(float value) {
  return setValue(std::vector<float>({value}));
}

Plot::Series &Plot::Series::setValue(float value_a, float value_b) {
  return setValue(std::vector<Point2>({{value_a, value_b}}));
}

const std::string &Plot::Series::label() const { return label_; }
bool Plot::Series::legend() const { return legend_; }
Color Plot::Series::color() const { return color_; }
bool Plot::Series::collides() const {
  return type_ == Histogram || type_ == Vistogram;
}

void Plot::Series::bounds(float &x_min, float &x_max, float &y_min,
                          float &y_max, int &n_max, int &p_max) const {
  for (const auto &e : entries_) {
    auto xe = e, xd = dims_, ye = e + dims_, yd = (type_ == Range ? 2 : 1);
    if (type_ == Vertical || type_ == Vistogram) {
      auto s = xe;
      xe = ye;
      ye = s;
      s = xd;
      xd = yd;
      yd = s;
    }
    if (type_ != Horizontal) {
      EXPECT_EQ(xd, 1);
      const auto &x = data_[xe];
      if (x_min > x) {
        x_min = x;
      }
      if (x_max < x) {
        x_max = x;
      }
    }
    if (type_ != Vertical) {
      for (auto yi = ye, _y = yi + yd; yi != _y; yi++) {
        const auto &y = data_[yi];
        if (y_min > y) {
          y_min = y;
        }
        if (y_max < y) {
          y_max = y;
        }
      }
    }
  }
  if (n_max < entries_.size()) {
    n_max = entries_.size();
  }
  if (type_ == Histogram || type_ == Vistogram) {
    p_max = std::max(30, p_max);
  }
}

void Plot::Series::dot(void *b, int x, int y, int r) const {
  auto &buffer = *(cv::Mat *)b;
  cv::circle(buffer, {x, y}, r, color2scalar(color_), -1, CV_AA);
}

void Plot::Series::draw(void *b, float x_min, float x_max, float y_min,
                        float y_max, float xs, float xd, float ys, float yd,
                        float x_axis, float y_axis, int unit,
                        float offset) const {
  if (dims_ == 0 || depth_ == 0) {
    return;
  }
  auto buffer = (cv::Mat *)b;
  cv::Mat alpha_buffer;
  if (color_.a != 255) {
    buffer->copyTo(alpha_buffer);
    buffer = &alpha_buffer;
  }
  auto color = color2scalar(color_);
  switch (type_) {
    case Line:
    case DotLine:
    case Dots: {
      EXPECT_DIMS_DEPTH(1, 1);
      bool has_last = false;
      float last_x, last_y;
      for (const auto &e : entries_) {
        auto x = data_[e], y = data_[e + dims_];
        cv::Point point((int)(x * xs + xd), (int)(y * ys + yd));
        if (has_last) {
          if (type_ == DotLine || type_ == Line) {
            cv::line(*buffer,
                     {(int)(last_x * xs + xd), (int)(last_y * ys + yd)}, point,
                     color, 1, CV_AA);
          }
        } else {
          has_last = true;
        }
        if (type_ == DotLine || type_ == Dots) {
          cv::circle(*buffer, point, 2, color, 1, CV_AA);
        }
        last_x = x, last_y = y;
      }
    } break;
    case Vistogram:
    case Histogram: {
      EXPECT_DIMS_DEPTH(1, 1);
      auto u = 2 * unit;
      auto o = (int)(2 * u * offset);
      for (const auto &e : entries_) {
        auto x = data_[e], y = data_[e + dims_];
        if (type_ == Histogram) {
          cv::rectangle(*buffer,
                        {(int)(x * xs + xd) - u + o, (int)(y_axis * ys + yd)},
                        {(int)(x * xs + xd) + u + o, (int)(y * ys + yd)}, color,
                        -1, CV_AA);
        } else if (type_ == Vistogram) {
          cv::rectangle(*buffer,
                        {(int)(x_axis * xs + xd), (int)(x * ys + yd) - u + o},
                        {(int)(y * xs + xd), (int)(x * ys + yd) + u + o}, color,
                        -1, CV_AA);
        }
      }

    } break;
    case Horizontal:
    case Vertical: {
      EXPECT_DIMS_DEPTH(1, 1);
      for (const auto &e : entries_) {
        auto y = data_[e + dims_];
        if (type_ == Horizontal) {
          cv::line(*buffer, {(int)(x_min * xs + xd), (int)(y * ys + yd)},
                   {(int)(x_max * xs + xd), (int)(y * ys + yd)}, color, 1,
                   CV_AA);
        } else if (type_ == Vertical) {
          cv::line(*buffer, {(int)(y * xs + xd), (int)(y_min * ys + yd)},
                   {(int)(y * xs + xd), (int)(y_max * ys + yd)}, color, 1,
                   CV_AA);
        }
      }
    } break;
    case Range: {
      EXPECT_DIMS_DEPTH(1, 2);
      bool has_last = false;
      cv::Point last_a, last_b;
      for (const auto &e : entries_) {
        auto x = data_[e], y_a = data_[e + dims_], y_b = data_[e + dims_ + 1];
        cv::Point point_a((int)(x * xs + xd), (int)(y_a * ys + yd));
        cv::Point point_b((int)(x * xs + xd), (int)(y_b * ys + yd));
        if (has_last) {
          cv::Point points[4] = {point_a, point_b, last_b, last_a};
          const cv::Point *p = points;
          auto count = 4;
          cv::fillPoly(*buffer, &p, &count, 1, color, CV_AA);
        } else {
          has_last = true;
        }
        last_a = point_a, last_b = point_b;
      }
    } break;
    case Circle: {
      EXPECT_DIMS_DEPTH(1, 2);
      for (const auto &e : entries_) {
        auto x = data_[e], y = data_[e + dims_], r = data_[e + dims_ + 1];
        cv::Point point((int)(x * xs + xd), (int)(y * ys + yd));
        cv::circle(*buffer, point, r, color, -1, CV_AA);
      }
    } break;
  }
  if (color_.a != 255) {
    addWeighted(*buffer, color_.a / 255.f, *(cv::Mat *)b, 1 - color_.a / 255.f,
                0, *(cv::Mat *)b);
  }
}

Plot::Figure &Plot::Figure::clear() {
  series_.clear();
  return *this;
}

Plot::Figure &Plot::Figure::origin(bool x, bool y) {
  include_zero_x_ = x, include_zero_y_ = y;
  return *this;
}

Plot::Figure &Plot::Figure::square(bool square) {
  aspect_square_ = square;
  return *this;
}

Plot::Figure &Plot::Figure::border(int size) {
  border_size_ = size;
  return *this;
}

Plot::Figure &Plot::Figure::window(const std::string &window) {
  window_ = window;
  return *this;
}

Plot::Series &Plot::Figure::series(const std::string &label) {
  for (auto &s : series_) {
    if (s.label() == label) {
      return s;
    }
  }
  Series s(label, Line, Color::hash(label));
  series_.push_back(s);
  return series_.back();
}

void Plot::Figure::draw(void *b, float x_min, float x_max, float y_min,
                        float y_max, int n_max, int p_max) const {
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
  for (const auto &s : series_) {
    if (s.collides()) {
      index++;
    }
  }
  std::max((int)series_.size() - 1, 1);
  for (auto s = series_.rbegin(); s != series_.rend(); ++s) {
    if (s->collides()) {
      index--;
    }
    s->draw(&buffer, x_min, x_max, y_min, y_max, xs, xd, ys, yd, x_axis, y_axis,
            unit, (float)index / series_.size());
  }

  // draw label names
  index = 0;
  for (const auto &s : series_) {
    if (!s.legend()) {
      continue;
    }
    auto name = s.label();
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
    s.dot(&buffer, buffer.cols - border_size_ - 10, org.y - 3, 3);
    index++;
  }
}

void Plot::Figure::show(bool flush) const {
  auto x_min = (include_zero_x_ ? 0.f : FLT_MAX);
  auto x_max = (include_zero_x_ ? 0.f : FLT_MIN);
  auto y_min = (include_zero_y_ ? 0.f : FLT_MAX);
  auto y_max = (include_zero_y_ ? 0.f : FLT_MIN);
  auto n_max = 0;
  auto p_max = grid_padding_;

  // find value bounds
  for (const auto &s : series_) {
    s.bounds(x_min, x_max, y_min, y_max, n_max, p_max);
  }

  if (n_max) {
    Rect rect(0, 0, 0, 0);
    auto &buffer = *(cv::Mat *)cvplot::buffer(window_.c_str(), rect.x, rect.y,
                                              rect.width, rect.height);
    auto sub = buffer({rect.x, rect.y, rect.width, rect.height});
    draw(&sub, x_min, x_max, y_min, y_max, n_max, p_max);
    cvplot::show(window_.c_str(), flush);
    // cvWaitKey(1);
  }
}

Plot::Figure &Plot::figure(const std::string &window) {
  if (figures_.count(window) == 0) {
    Figure figure;
    figure.window(window);
    figures_[window] = figure;
  }
  return figures_[window];
}

Plot::Figure &figure(const std::string &window) { return Plot::shared(window); }

}  // namespace cvplot
