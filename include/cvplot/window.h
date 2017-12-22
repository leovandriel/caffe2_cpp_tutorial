#ifndef CVPLOT_WINDOW_H
#define CVPLOT_WINDOW_H

#include "color.h"

#include <map>

namespace cvplot {

struct Rect {
  int x, y, width, height;
  Rect(int x, int y, int width, int height)
      : x(x), y(y), width(width), height(height) {}
};

struct Size {
  int width, height;
  Size(int width, int height) : width(width), height(height) {}
};

struct Offset {
  int x, y;
  Offset(int x, int y) : x(x), y(y) {}
};

class Window;

class View {
 public:
  View(Window &window, const std::string &title = "", Size size = {300, 300})
      : window_(window),
        title_(title),
        rect_(0, 0, size.width, size.height),
        frameless_(false),
        background_color_(Black),
        frame_color_(Green),
        text_color_(Black) {}
  View &resize(Rect rect);
  View &size(Size size);
  View &offset(Offset offset);
  View &autosize();
  View &title(const std::string &title);
  View &alpha(int alpha);
  View &backgroundColor(Color color);
  View &frameColor(Color color);
  View &textColor(Color color);
  Color backgroundColor();
  Color frameColor();
  Color textColor();

  void drawFill(Color background = White);
  void drawImage(const void *image, int alpha = 255);
  void drawText(const std::string &text, Offset offset, Color color) const;
  void drawFrame(const std::string &title) const;
  void *buffer(Rect &rect);
  void show(bool flush = true) const;

 protected:
  Rect rect_;
  std::string title_;
  bool frameless_;
  Window &window_;
  Color background_color_;
  Color frame_color_;
  Color text_color_;
};

class Window {
 public:
  Window(const std::string &title = "");
  Window &resize(Rect rect, bool flush = true);
  Window &size(Size size, bool flush = true);
  Window &offset(Offset offset);
  Window &title(const std::string &title);
  Window &ensure(Rect rect, bool flush = true);
  void show(bool flush = true) const;
  void *buffer();
  View &view(const std::string &name, Size size = {300, 300});

 protected:
  Offset offset_;
  void *buffer_;
  std::string title_;
  std::string name_;
  std::map<std::string, View> views_;
};

void window(const char *title, int width = 0, int height = 0,
            bool flush = false);
View &view(const char *name);
void move(int x, int y);
void move(const char *name, int x, int y);
void resize(const char *name, int width, int height);
void clear(const char *name, bool flush = false);
void autosize(const char *name);
void title(const char *name, const char *title);
void imshow(const char *name, const void *image, bool flush = true);
void *buffer(const char *name, int &x, int &y, int &width, int &height);
void show(const char *name, bool flush = true);
void show(bool flush = true);
Window &window();

}  // namespace cvplot

#endif  // CVPLOT_WINDOW_H
