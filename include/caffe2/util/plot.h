#ifndef UTIL_PLOT_H
#define UTIL_PLOT_H

#include <map>
#include <string>
#include <vector>

namespace caffe2 {

const int paleness = 32;

class PlotUtil {
 public:
  enum Type { Line, DotLine, Dots, Histogram, Vistogram, Horizontal, Vertical };

  struct Color {
    int r, g, b;
    Color(int r, int g, int b) : r(r), g(g), b(b) {}
    Color(int v) : Color(v, v, v) {}
    Color(int v[3]) : Color(v[0], v[1], v[2]) {}
    Color() : Color(0) {}
    Color(float hue);
    Color(const std::string &seed)
        : Color(std::hash<std::string>{}(seed) % 600 / 100.f) {}
  };

  static Color Red() { return Color(0.f); }
  static Color Orange() { return Color(.5f); }
  static Color Yellow() { return Color(1.f); }
  static Color Lawn() { return Color(1.5f); }
  static Color Green() { return Color(2.f); }
  static Color Aqua() { return Color(2.5f); }
  static Color Cyan() { return Color(3.f); }
  static Color Sky() { return Color(3.5f); }
  static Color Blue() { return Color(4.f); }
  static Color Purple() { return Color(4.5f); }
  static Color Magenta() { return Color(5.f); }
  static Color Pink() { return Color(5.5f); }
  static Color Black() { return Color(paleness); }
  static Color Dark() { return Color(paleness * 2); }
  static Color Gray() { return Color(128); }
  static Color Light() { return Color(256 - paleness * 2); }
  static Color White() { return Color(256 - paleness); }

  struct Series {
    Series() {}
    Series(const std::string &label, enum Type type, Color color)
        : label_(label), type_(type), color_(color) {}

    void Clear() { data_.clear(); }
    void Type(enum Type type) { type_ = type; }
    void Color(Color color) { color_ = color; }

    void Append(const std::vector<std::pair<float, float>> &data) {
      data_.insert(data_.end(), data.begin(), data.end());
    }

    void Append(const std::vector<float> &values) {
      std::vector<std::pair<float, float>> data(values.size());
      auto i = 0;
      for (auto &d : data) {
        d.first = i + data_.size();
        d.second = values[i++];
      }
      Append(data);
    }
    void Append(float key, float value) {
      Append(std::vector<std::pair<float, float>>({{key, value}}));
    }
    void Append(float value) { Append(std::vector<float>({value})); }
    void Set(const std::vector<std::pair<float, float>> &data, enum Type type,
             struct Color color) {
      Clear();
      Type(type);
      Color(color);
      Append(data);
    }

    void Set(const std::vector<float> &values, enum Type type,
             struct Color color) {
      std::vector<std::pair<float, float>> data(values.size());
      auto i = 0;
      for (auto &d : data) {
        d.first = i;
        d.second = values[i++];
      }
      Set(data, type, color);
    }
    void Set(float key, float value, enum Type type, struct Color color) {
      Set(std::vector<std::pair<float, float>>({{key, value}}), type, color);
    }
    void Set(float value, enum Type type, struct Color color) {
      Set(std::vector<float>({value}), type, color);
    }

    std::string &Label() { return label_; }
    bool Collides() { return type_ == Histogram || type_ == Vistogram; }

    void Bounds(float &x_min, float &x_max, float &y_min, float &y_max,
                int &n_max, int &p_max);
    void Dot(void *b, int x, int y, int r);
    void Draw(void *buffer, float x_min, float x_max, float y_min, float y_max,
              float x_axis, float xs, float xd, float ys, float yd,
              float y_axis, int unit, float offset);

   protected:
    std::vector<std::pair<float, float>> data_;
    enum Type type_;
    struct Color color_;
    std::string label_;
  };

  class Figure {
   public:
    Figure()
        : border_size_(50),
          background_color_(White()),
          axis_color_(Black()),
          sub_axis_color_(Light()),
          text_color_(Black()),
          include_zero_x_(true),
          include_zero_y_(true),
          aspect_square_(false),
          grid_size_(60),
          grid_padding_(20) {}

    void Clear() { series_.clear(); }
    void Origin(bool x, bool y) { include_zero_x_ = x, include_zero_y_ = y; }
    void Square(bool square) { aspect_square_ = square; }
    void Window(const std::string &window) { window_ = window; }

    void Draw(void *b, float x_min, float x_max, float y_min, float y_max,
              int n_max, int p_max);
    void Show();

    Series &Get(const std::string &label) {
      for (auto &s : series_) {
        if (s.Label() == label) {
          return s;
        }
      }
      series_.push_back(Series(label, Line, Color(label)));

      return series_.back();
    }

   protected:
    std::string window_;
    std::vector<Series> series_;
    int border_size_;
    Color background_color_;
    Color axis_color_;
    Color sub_axis_color_;
    Color text_color_;
    bool include_zero_x_;
    bool include_zero_y_;
    bool aspect_square_;
    int grid_size_;
    int grid_padding_;
  };

  Figure &Get(const std::string &window) {
    if (figures_.count(window) == 0) {
      auto figure = Figure();
      figure.Window(window);
      figures_[window] = figure;
    }
    return figures_[window];
  }

  static Figure &Shared(const std::string &window);

 protected:
  std::map<std::string, Figure> figures_;
};

}  // namespace caffe2

#endif  // UTIL_PLOT_H
