#ifndef UTIL_PLOT_H
#define UTIL_PLOT_H

#include <map>
#include <string>
#include <vector>

namespace caffe2 {

const int paleness = 32;

class PlotUtil {
 public:
  enum Type {
    Line,
    DotLine,
    Dots,
    Histogram,
    Vistogram,
    Horizontal,
    Vertical,
    Range,
    Circle,
  };

  struct Color {
    uint8_t r, g, b, a;
    Color(uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255)
        : r(r), g(g), b(b), a(a) {}
    Color(const uint8_t *rgb, uint8_t a = 255)
        : Color(rgb[0], rgb[1], rgb[2], a) {}
    Color() : Color(0, 0, 0) {}

    Color &Alpha(uint8_t alpha) {
      a = alpha;
      return *this;
    }

    static Color Gray(uint8_t v) { return Color(v, v, v); }
    static Color Hue(float hue);
    static Color Cos(float hue);
    static Color Index(uint32_t index, uint32_t density = 16, float avoid = 2.f,
                       float range = 2.f) {  // avoid greens by default
      if (avoid > 0) {
        auto step = density / (6 - range);
        auto offset = (avoid + range / 2) * step;
        index = offset + index % density;
        density += step * range;
      }
      auto hue = index % density * 6.f / density;
      return Color::Cos(hue);
    }
    static Color Hash(const std::string &seed) {
      return Color::Index(std::hash<std::string>{}(seed));
    }
  };

  static Color Red() { return Color::Hue(0.f); }
  static Color Orange() { return Color::Hue(.5f); }
  static Color Yellow() { return Color::Hue(1.f); }
  static Color Lawn() { return Color::Hue(1.5f); }
  static Color Green() { return Color::Hue(2.f); }
  static Color Aqua() { return Color::Hue(2.5f); }
  static Color Cyan() { return Color::Hue(3.f); }
  static Color Sky() { return Color::Hue(3.5f); }
  static Color Blue() { return Color::Hue(4.f); }
  static Color Purple() { return Color::Hue(4.5f); }
  static Color Magenta() { return Color::Hue(5.f); }
  static Color Pink() { return Color::Hue(5.5f); }
  static Color Black() { return Color::Gray(paleness); }
  static Color Dark() { return Color::Gray(paleness * 2); }
  static Color Gray() { return Color::Gray(128); }
  static Color Light() { return Color::Gray(256 - paleness * 2); }
  static Color White() { return Color::Gray(256 - paleness); }

  struct Point2 {
    float x, y;
  };

  struct Series {
    Series(const std::string &label, enum Type type, Color color)
        : label_(label),
          type_(type),
          color_(color),
          dims_(0),
          depth_(0),
          legend_(true) {}

    void Clear() {
      entries_.clear();
      data_.clear();
    }
    Series &Type(enum Type type) {
      type_ = type;
      return *this;
    }
    Series &Color(Color color) {
      color_ = color;
      return *this;
    }
    Series &Dims(int dims);
    Series &Depth(int depth);
    Series &Legend(bool legend) {
      legend_ = legend;
      return *this;
    }

    void Ensure(int dims, int depth);

    Series &Add(const std::vector<std::pair<float, float>> &data) {
      Ensure(1, 1);
      for (const auto &d : data) {
        entries_.push_back(data_.size());
        data_.push_back(d.first);
        data_.push_back(d.second);
      }
      return *this;
    }

    Series &Add(const std::vector<std::pair<float, Point2>> &data) {
      Ensure(1, 2);
      for (const auto &d : data) {
        entries_.push_back(data_.size());
        data_.push_back(d.first);
        data_.push_back(d.second.x);
        data_.push_back(d.second.y);
      }
      return *this;
    }

    Series &AddValue(const std::vector<float> &values) {
      std::vector<std::pair<float, float>> data(values.size());
      auto i = 0;
      for (auto &d : data) {
        d.first = i + entries_.size();
        d.second = values[i++];
      }
      return Add(data);
    }
    Series &AddValue(const std::vector<Point2> &values) {
      std::vector<std::pair<float, Point2>> data(values.size());
      auto i = 0;
      for (auto &d : data) {
        d.first = i + entries_.size();
        d.second = values[i++];
      }
      return Add(data);
    }

    Series &Add(float key, float value) {
      return Add(std::vector<std::pair<float, float>>({{key, value}}));
    }
    Series &Add(float key, Point2 value) {
      return Add(std::vector<std::pair<float, Point2>>({{key, value}}));
    }

    Series &AddValue(float value) {
      return AddValue(std::vector<float>({value}));
    }
    Series &AddValue(float value_a, float value_b) {
      return AddValue(std::vector<Point2>({{value_a, value_b}}));
    }

    Series &Set(const std::vector<std::pair<float, float>> &data) {
      Dims(1).Depth(1);
      Clear();
      return Add(data);
    }
    Series &Set(const std::vector<std::pair<float, Point2>> &data) {
      Dims(1).Depth(2);
      Clear();
      return Add(data);
    }

    Series &SetValue(const std::vector<float> &values) {
      std::vector<std::pair<float, float>> data(values.size());
      auto i = 0;
      for (auto &d : data) {
        d.first = i;
        d.second = values[i++];
      }
      return Set(data);
    }
    Series &SetValue(const std::vector<Point2> &values) {
      std::vector<std::pair<float, Point2>> data(values.size());
      auto i = 0;
      for (auto &d : data) {
        d.first = i;
        d.second = values[i++];
      }
      return Set(data);
    }

    Series &Set(float key, float value) {
      return Set(std::vector<std::pair<float, float>>({{key, value}}));
    }
    Series &Set(float key, float value_a, float value_b) {
      return Set(
          std::vector<std::pair<float, Point2>>({{key, {value_a, value_b}}}));
    }

    Series &SetValue(float value) {
      return SetValue(std::vector<float>({value}));
    }
    Series &SetValue(float value_a, float value_b) {
      return SetValue(std::vector<Point2>({{value_a, value_b}}));
    }

    std::string &Label() { return label_; }
    bool Legend() { return legend_; }
    struct Color Color() {
      return color_;
    }
    bool Collides() { return type_ == Histogram || type_ == Vistogram; }

    void Bounds(float &x_min, float &x_max, float &y_min, float &y_max,
                int &n_max, int &p_max);
    void Dot(void *b, int x, int y, int r);
    void Draw(void *buffer, float x_min, float x_max, float y_min, float y_max,
              float x_axis, float xs, float xd, float ys, float yd,
              float y_axis, int unit, float offset);

   protected:
    std::vector<int> entries_;
    std::vector<float> data_;
    enum Type type_;
    struct Color color_;
    std::string label_;
    int dims_;
    int depth_;
    bool legend_;
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
    Figure &Origin(bool x, bool y) {
      include_zero_x_ = x, include_zero_y_ = y;
      return *this;
    }
    Figure &Square(bool square) {
      aspect_square_ = square;
      return *this;
    }
    Figure &Border(int size) {
      border_size_ = size;
      return *this;
    }
    Figure &Window(const std::string &window) {
      window_ = window;
      return *this;
    }

    void Draw(void *b, float x_min, float x_max, float y_min, float y_max,
              int n_max, int p_max);
    void Show(bool flush = true);

    Series &Get(const std::string &label) {
      for (auto &s : series_) {
        if (s.Label() == label) {
          return s;
        }
      }
      series_.push_back(Series(label, Line, Color::Hash(label)));

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
