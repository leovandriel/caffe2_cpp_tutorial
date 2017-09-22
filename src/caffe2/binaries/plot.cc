#include "caffe2/util/plot.h"
#include "caffe2/core/init.h"
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
    figure.Get("line").Set({1.f, 2.f, 3.f, 4.f, 5.f}, PlotUtil::DotLine,
                           PlotUtil::Blue());
    figure.Get("histogram")
        .Set({1.f, 2.f, 3.f, 4.f, 5.f}, PlotUtil::Histogram, PlotUtil::Red());
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
    figure.Get("parabola").Set(values, PlotUtil::DotLine, PlotUtil::Green());
    values.clear();
    for (auto i = 0; i <= 10; i++) {
      values.push_back(sin(i / 1.5f) * 5);
    }
    figure.Get("sine").Set(values, PlotUtil::DotLine, PlotUtil::Blue());
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
    figure.Get("uniform").Set(data, PlotUtil::Dots, PlotUtil::Orange());
    data.clear();
    for (auto i = 0; i <= 100; i++) {
      data.push_back(
          {exp((rand() % 100) / 30.f) - 1, exp((rand() % 100) / 30.f) - 1});
    }
    figure.Get("exponential").Set(data, PlotUtil::Dots, PlotUtil::Sky());
    figure.Show();
  }

  {
    auto name = "histograms";
    setWindowTitle(name, "multiple histograms");
    moveWindow(name, 0, 300);
    resizeWindow(name, 300, 300);
    auto &figure = PlotUtil::Shared(name);
    figure.Get("1").Set({1.f, 2.f, 3.f, 4.f, 5.f}, PlotUtil::Histogram,
                        PlotUtil::Blue());
    figure.Get("2").Set({6.f, 5.f, 4.f, 3.f, 2.f, 1.f}, PlotUtil::Histogram,
                        PlotUtil::Green());
    figure.Get("3").Set({3.f, 1.f, -1.f, 1.f, 3.f, 7.f}, PlotUtil::Histogram,
                        PlotUtil::Red());
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
    figure.Get("circle").Set(data, PlotUtil::Line, PlotUtil::Purple());
    data.clear();
    for (auto i = 0; i <= 100; i++) {
      data.push_back({cos(i * .2513f + 1), sin(i * .0628f + 4)});
    }
    figure.Get("lissajous").Set(data, PlotUtil::Line, PlotUtil::Aqua());
    figure.Show();
  }

  {
    auto name = "non zero";
    setWindowTitle(name, "non-zero axis");
    moveWindow(name, 600, 600);
    resizeWindow(name, 300, 300);
    auto &figure = PlotUtil::Shared(name);
    figure.Origin(false, false);
    figure.Get("histogram")
        .Set({4.f, 5.f, 7.f}, PlotUtil::Histogram, PlotUtil::Blue());
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
    figure.Get("random").Set(PlotUtil::Line, PlotUtil::Gray());
    for (int i = 0; i < 2000; i++) {
      auto l = sqrt((dx * dx + dy * dy) * (f * f + 1)) * 10;
      dx = (dx + f * dy) / l;
      dy = (dy - f * dx) / l;
      f = (f + df) * 0.8f;
      df = (df + rand() % 11 / 100.f - .05f) * 0.8f;
      figure.Get("random").Append(x += dx, y += dy);
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
