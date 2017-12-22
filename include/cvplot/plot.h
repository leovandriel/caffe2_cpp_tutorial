#ifndef CVPLOT_PLOT_H
#define CVPLOT_PLOT_H

#include "color.h"

#include <map>
#include <string>
#include <vector>

namespace cvplot {

struct Point2 {
  float x, y;
};

class Plot {
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

  class Series {
   public:
    Series(const std::string &label, enum Type type, Color color)
        : label_(label),
          type_(type),
          color_(color),
          dims_(0),
          depth_(0),
          legend_(true) {}

    Series &type(enum Type type);
    Series &color(Color color);
    Series &dims(int dims);
    Series &depth(int depth);
    Series &legend(bool legend);
    Series &add(const std::vector<std::pair<float, float>> &data);
    Series &add(const std::vector<std::pair<float, Point2>> &data);
    Series &addValue(const std::vector<float> &values);
    Series &addValue(const std::vector<Point2> &values);
    Series &add(float key, float value);
    Series &add(float key, Point2 value);
    Series &addValue(float value);
    Series &addValue(float value_a, float value_b);
    Series &set(const std::vector<std::pair<float, float>> &data);
    Series &set(const std::vector<std::pair<float, Point2>> &data);
    Series &setValue(const std::vector<float> &values);
    Series &setValue(const std::vector<Point2> &values);
    Series &set(float key, float value);
    Series &set(float key, float value_a, float value_b);
    Series &setValue(float value);
    Series &setValue(float value_a, float value_b);
    Series &clear();

    const std::string &label() const;
    bool legend() const;
    Color color() const;
    bool collides() const;
    void ensure(int dims, int depth);
    void bounds(float &x_min, float &x_max, float &y_min, float &y_max,
                int &n_max, int &p_max) const;
    void dot(void *b, int x, int y, int r) const;
    void draw(void *buffer, float x_min, float x_max, float y_min, float y_max,
              float x_axis, float xs, float xd, float ys, float yd,
              float y_axis, int unit, float offset) const;

   protected:
    std::vector<int> entries_;
    std::vector<float> data_;
    enum Type type_;
    Color color_;
    std::string label_;
    int dims_;
    int depth_;
    bool legend_;
  };

  class Figure {
   public:
    Figure()
        : border_size_(50),
          background_color_(White),
          axis_color_(Black),
          sub_axis_color_(Light),
          text_color_(Black),
          include_zero_x_(true),
          include_zero_y_(true),
          aspect_square_(false),
          grid_size_(60),
          grid_padding_(20) {}

    Figure &clear();
    Figure &origin(bool x, bool y);
    Figure &square(bool square);
    Figure &border(int size);
    Figure &window(const std::string &window);

    void draw(void *b, float x_min, float x_max, float y_min, float y_max,
              int n_max, int p_max) const;
    void show(bool flush = true) const;
    Series &series(const std::string &label);

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

  Figure &figure(const std::string &window);
  static Figure &shared(const std::string &window);

 protected:
  std::map<std::string, Figure> figures_;
};

Plot::Figure &figure(const std::string &window);

}  // namespace cvplot

#endif  // CVPLOT_PLOT_H
