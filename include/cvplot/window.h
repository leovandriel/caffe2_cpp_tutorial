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

class Window {
  class View {
   public:
    View(Window &window, const std::string &title, Size size)
        : window_(window),
          title_(title),
          rect_(0, 0, size.width, size.height),
          frameless_(false) {}
    void resize(Rect rect);
    void size(Size size);
    void offset(Offset offset);
    void autosize();
    void title(const std::string &title);

    void drawFill(Color background = {224, 224, 224});
    void drawImage(const void *image);
    void drawText(const std::string &text, Offset offset) const;
    void drawFrame(const std::string &title, Color foreground = {32, 224, 32},
                   Color background = {32, 32, 32},
                   Color text = {0, 0, 0}) const;
    void *buffer(Rect &rect);
    void show(bool flush = true) const;

   protected:
    Rect rect_;
    std::string title_;
    bool frameless_;
    Window &window_;
  };

 public:
  Window() : offset_(0, 0), buffer_(NULL) {}
  void resize(Rect rect, bool flush = true);
  void size(Size size, bool flush = true);
  void offset(Offset offset);
  void ensure(Rect rect, bool flush = true);
  void title(const std::string &title);
  void show(bool flush = true) const;
  View &view(const std::string &name, Size size = {300, 300});

 protected:
  Offset offset_;
  void *buffer_;
  std::string title_;
  std::map<std::string, View> views_;
};

void window(const char *title, int width = 0, int height = 0,
            bool flush = false);
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
