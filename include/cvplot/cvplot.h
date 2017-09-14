// Matlab style plot functions for OpenCV by Changbo (zoccob@gmail).
// plot and label:
//
// template<typename T>
// void plot(const std::string figure_name, const T* p, int count, int step = 1,
//		     int R = -1, int G = -1, int B = -1);
//
// figure_name: required. multiple calls of this function with same figure_name
//              plots multiple curves on a single graph.
// p          : required. pointer to data.
// count      : required. number of data.
// step       : optional. step between data of two points, default 1.
// R, G,B     : optional. assign a color to the curve.
//              if not assigned, the curve will be assigned a unique color
//              automatically.
//
// void label(std::string lbl):
//
// label the most recently added curve with lbl.
//
//
//
//

#pragma once

#if WIN32
#define snprintf sprintf_s
#endif

#include <vector>
#include "cv.h"
#include "highgui.h"

// using namespace std;

namespace CvPlot {
// A curve.
class Series {
 public:
  // number of points
  unsigned int count;
  float *data;
  // name of the curve
  std::string label;

  // allow automatic curve color
  bool auto_color;
  CvScalar color;

  Series(void);
  Series(const Series &s);
  ~Series(void);

  // release memory
  void Clear();

  void SetData(int n, float *p);

  void SetColor(CvScalar color, bool auto_color = true);
  void SetColor(int R, int G, int B, bool auto_color = true);
};

// a figure comprises of several curves
class Figure {
 private:
  // window name
  std::string figure_name;
  CvSize figure_size;

  // margin size
  int border_size;

  CvScalar backgroud_color;
  CvScalar axis_color;
  CvScalar text_color;

  // several curves
  std::vector<Series> plots;

  // manual or automatic range
  bool custom_range_y;
  float y_max;
  float y_min;

  float y_scale;

  bool custom_range_x;
  float x_max;
  float x_min;

  float x_scale;

  // automatically change color for each curve
  int color_index;

 public:
  Figure(const std::string name);
  ~Figure();

  std::string GetFigureName();
  Series *Add(const Series &s);
  void Clear();
  void DrawLabels(IplImage *output, int posx, int posy);

  // show plot window
  void Show();

 private:
  Figure();
  void DrawAxis(IplImage *output);
  void DrawPlots(IplImage *output);

  // call before plot
  void Initialize();
  CvScalar GetAutoColor();
};

// manage plot windows
class PlotManager {
 private:
  std::vector<Figure> figure_list;
  Series *active_series;
  Figure *active_figure;

 public:
  // now useless
  bool HasFigure(std::string wnd);
  Figure *FindFigure(std::string wnd);

  void Plot(const std::string figure_name, const float *p, int count, int step,
            int R, int G, int B);

  void Label(std::string lbl);
};

// handle different data types; static mathods;

template <typename T>
void plot(const std::string figure_name, const T *p, int count, int step = 1,
          int R = -1, int G = -1, int B = -1);
void clear(const std::string figure_name);

void label(std::string lbl);

};  // namespace CvPlot
