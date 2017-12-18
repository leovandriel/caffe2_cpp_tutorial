#include "caffe2/util/plot.h"
#include <caffe2/core/init.h>
#include "caffe2/util/window.h"

namespace caffe2 {

void run() {
  std::cout << "Plot Examples" << std::endl;

  std::vector<std::pair<float, float>> data;
  std::vector<float> values;

  superWindow("Plot Examples");

  {
    auto name = "simple";
    setWindowTitle(name, "line and histogram");
    moveWindow(name, 0, 0);
    resizeWindow(name, 300, 300);
    auto &figure = PlotUtil::Shared(name);
    figure.Get("line")
        .SetValue({1.f, 2.f, 3.f, 4.f, 5.f})
        .Type(PlotUtil::DotLine)
        .Color(PlotUtil::Blue());
    figure.Get("histogram")
        .SetValue({1.f, 2.f, 3.f, 4.f, 5.f})
        .Type(PlotUtil::Histogram)
        .Color(PlotUtil::Red());
    figure.Show();
  }

  {
    auto name = "math";
    setWindowTitle(name, "some curves");
    moveWindow(name, 300, 0);
    resizeWindow(name, 300, 300);
    auto &figure = PlotUtil::Shared(name);
    values.clear();
    for (auto i = 0; i <= 10; i++) {
      values.push_back((i - 4) * (i - 4) - 6);
    }
    figure.Get("parabola")
        .SetValue(values)
        .Type(PlotUtil::DotLine)
        .Color(PlotUtil::Green());
    values.clear();
    for (auto i = 0; i <= 10; i++) {
      values.push_back(sin(i / 1.5f) * 5);
    }
    figure.Get("sine")
        .SetValue(values)
        .Type(PlotUtil::DotLine)
        .Color(PlotUtil::Blue());
    values.clear();
    values.push_back(15);
    figure.Get("threshold")
        .SetValue(values)
        .Type(PlotUtil::Horizontal)
        .Color(PlotUtil::Red());
    figure.Show();
  }

  {
    auto name = "scatter";
    setWindowTitle(name, "scatter plots");
    moveWindow(name, 600, 0);
    resizeWindow(name, 300, 300);
    auto &figure = PlotUtil::Shared(name);
    data.clear();
    for (auto i = 0; i <= 100; i++) {
      data.push_back({(rand() % 100) / 10.f, (rand() % 100) / 10.f});
    }
    figure.Get("uniform")
        .Set(data)
        .Type(PlotUtil::Dots)
        .Color(PlotUtil::Orange());
    data.clear();
    for (auto i = 0; i <= 100; i++) {
      data.push_back(
          {exp((rand() % 100) / 30.f) - 1, exp((rand() % 100) / 30.f) - 1});
    }
    figure.Get("exponential")
        .Set(data)
        .Type(PlotUtil::Dots)
        .Color(PlotUtil::Sky());
    figure.Show();
  }

  {
    auto name = "histograms";
    setWindowTitle(name, "multiple histograms");
    moveWindow(name, 0, 300);
    resizeWindow(name, 300, 300);
    auto &figure = PlotUtil::Shared(name);
    figure.Get("1")
        .SetValue({1.f, 2.f, 3.f, 4.f, 5.f})
        .Type(PlotUtil::Histogram)
        .Color(PlotUtil::Blue());
    figure.Get("2")
        .SetValue({6.f, 5.f, 4.f, 3.f, 2.f, 1.f})
        .Type(PlotUtil::Histogram)
        .Color(PlotUtil::Green());
    figure.Get("3")
        .SetValue({3.f, 1.f, -1.f, 1.f, 3.f, 7.f})
        .Type(PlotUtil::Histogram)
        .Color(PlotUtil::Red());
    figure.Show();
  }

  {
    auto name = "parametric";
    setWindowTitle(name, "parametric plots");
    moveWindow(name, 300, 300);
    resizeWindow(name, 600, 300);
    auto &figure = PlotUtil::Shared(name);
    figure.Square(true);
    data.clear();
    for (auto i = 0; i <= 100; i++) {
      data.push_back({cos(i * .0628f + 4) * 2, sin(i * .0628f + 4) * 2});
    }
    figure.Get("circle").Add(data);
    data.clear();
    for (auto i = 0; i <= 100; i++) {
      data.push_back({cos(i * .2513f + 1), sin(i * .0628f + 4)});
    }
    figure.Get("lissajous").Add(data);
    figure.Show();
  }

  {
    auto name = "no-axis";
    setWindowTitle(name, "hidden axis");
    moveWindow(name, 600, 600);
    resizeWindow(name, 300, 300);
    auto &figure = PlotUtil::Shared(name);
    figure.Origin(false, false);
    figure.Get("histogram")
        .SetValue({4.f, 5.f, 7.f, 6.f})
        .Type(PlotUtil::Vistogram)
        .Color(PlotUtil::Blue());
    figure.Get("min")
        .SetValue(4.f)
        .Type(PlotUtil::Vertical)
        .Color(PlotUtil::Pink());
    figure.Get("max")
        .SetValue(7.f)
        .Type(PlotUtil::Vertical)
        .Color(PlotUtil::Purple());
    figure.Show();
  }

  {
    auto name = "colors";
    setWindowTitle(name, "auto color");
    moveWindow(name, 900, 0);
    resizeWindow(name, 300, 300);
    auto &figure = PlotUtil::Shared(name);
    for (auto i = 0; i < 16; i++) {
      figure.Get(std::to_string(i))
          .Set(2, i + 1)
          .Type(PlotUtil::Vistogram)
          .Color(PlotUtil::Color::Index(i));
    }
    figure.Show();
  }

  {
    auto name = "range";
    setWindowTitle(name, "range plot");
    moveWindow(name, 900, 300);
    resizeWindow(name, 300, 300);
    auto &figure = PlotUtil::Shared(name);
    values.clear();
    figure.Get("apples").Type(PlotUtil::Line).Color(PlotUtil::Orange());
    figure.Get("pears").Type(PlotUtil::Line).Color(PlotUtil::Sky());
    figure.Get("apples_range")
        .Type(PlotUtil::Range)
        .Color(PlotUtil::Orange().Alpha(128))
        .Depth(2)
        .Legend(false);
    figure.Get("pears_range")
        .Type(PlotUtil::Range)
        .Color(PlotUtil::Sky().Alpha(128))
        .Depth(2)
        .Legend(false);
    for (auto i = 0; i <= 10; i++) {
      float v = (i - 4) * (i - 4) - 6;
      figure.Get("apples_range")
          .AddValue(v + 5.f * rand() / RAND_MAX,
                    v + 20.f + 5.f * rand() / RAND_MAX);
      figure.Get("apples").AddValue(v + 10.f + 5.f * rand() / RAND_MAX);
      v = -(i - 6) * (i - 6) + 30;
      figure.Get("pears_range")
          .AddValue(v + 5.f * rand() / RAND_MAX,
                    v + 20.f + 5.f * rand() / RAND_MAX);
      figure.Get("pears").AddValue(v + 10.f + 5.f * rand() / RAND_MAX);
    }
    figure.Show();
  }

  {
    auto name = "balls";
    setWindowTitle(name, "balls");
    moveWindow(name, 900, 600);
    resizeWindow(name, 300, 300);
    auto &figure = PlotUtil::Shared(name);
    figure.Get("purple")
        .Type(PlotUtil::Circle)
        .Color(PlotUtil::Purple().Alpha(128))
        .Depth(2);
    figure.Get("yellow")
        .Type(PlotUtil::Circle)
        .Color(PlotUtil::Yellow().Alpha(200))
        .Depth(2);
    for (auto i = 0; i <= 20; i++) {
      figure.Get("purple").Add((rand() % 100) / 10.f,
                               {(rand() % 100) / 10.f, (rand() % 100) / 5.f});
      figure.Get("yellow").Add((rand() % 100) / 10.f,
                               {(rand() % 100) / 10.f, (rand() % 100) / 5.f});
    }
    figure.Show();
  }

  {
    auto name = "dynamic";
    setWindowTitle(name, "dynamic plotting");
    moveWindow(name, 0, 600);
    resizeWindow(name, 600, 300);
    auto &figure = PlotUtil::Shared(name);
    figure.Square(true);
    figure.Origin(false, false);
    srand(clock());
    auto x = 0.f, y = 0.f, dx = 1.f, dy = 0.f, f = 0.f, df = 0.f;
    for (int i = 0; i < 2000; i++) {
      auto l = sqrt((dx * dx + dy * dy) * (f * f + 1)) * 10;
      dx = (dx + f * dy) / l;
      dy = (dy - f * dx) / l;
      f = (f + df) * 0.8f;
      df = (df + rand() % 11 / 100.f - .05f) * 0.8f;
      figure.Get("random").Add(x += dx, y += dy);
      figure.Get("random").Color(PlotUtil::Color::Index(i));
      figure.Show();
      cvWaitKey(10);
    }
  }
}

}  // namespace caffe2

int main(int argc, char **argv) {
  caffe2::GlobalInit(&argc, &argv);
  caffe2::run();
  google::protobuf::ShutdownProtobufLibrary();
  return 0;
}
