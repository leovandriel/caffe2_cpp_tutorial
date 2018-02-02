#ifndef ZOO_MOBILENET_H
#define ZOO_MOBILENET_H

#include "caffe2/util/model.h"

namespace caffe2 {

std::string tos2(int i) { return std::to_string(i); }

class MobileNetModel : public ModelUtil {
 public:
  MobileNetModel(NetDef &initnet, NetDef &predictnet)
      : ModelUtil(initnet, predictnet) {}

  OperatorDef *AddConvOps(const std::string &input, const std::string &output,
                          const std::string &infix, int in_size, int out_size,
                          int stride, int padding, int kernel, bool group,
                          bool train) {
    auto b = output + (infix.size() ? "_conv_" + infix : "");
    init.AddMSRAFillOp({out_size, (group ? 1 : in_size), kernel, kernel},
                       b + "_w");
    predict.AddInput(b + "_w");
    auto op = predict.AddConvOp(input, b + "_w", "", b, stride, padding, kernel,
                                (group ? in_size : 0));
    auto p = output + "_spatbn" + (infix.size() ? "_" + infix : "");
    init.AddConstantFillOp({out_size}, 1.f, p + "_s");
    predict.AddInput(p + "_s");
    init.AddConstantFillOp({out_size}, 0.f, p + "_b");
    predict.AddInput(p + "_b");
    init.AddConstantFillOp({out_size}, 0.f, p + "_rm");
    predict.AddInput(p + "_rm");
    init.AddConstantFillOp({out_size}, 1.f, p + "_riv");
    predict.AddInput(p + "_riv");
    return predict.AddSpatialBNOp(
        {b, p + "_s", p + "_b", p + "_rm", p + "_riv"},
        train ? std::vector<std::string>(
                    {p, p + "_rm", p + "_riv", p + "_sm", p + "_siv"})
              : std::vector<std::string>({p}),
        0.001, 0.9, !train);
  }

  OperatorDef *AddFcOps(const std::string &input, const std::string &output,
                        int in_size, int out_size) {
    init.AddXavierFillOp({out_size, in_size}, output + "_w");
    predict.AddInput(output + "_w");
    init.AddConstantFillOp({out_size}, output + "_b");
    predict.AddInput(output + "_b");
    return predict.AddFcOp(input, output + "_w", output + "_b", output);
  }

  OperatorDef *AddFirst(const std::string &prefix, const std::string &input,
                        int out_size, int stride, float alpha, bool train) {
    auto output = "conv" + prefix;
    std::string layer = input;
    layer = AddConvOps(layer, output, "", 3, out_size * alpha, stride, 1, 3,
                       false, train)
                ->output(0);
    return predict.AddReluOp(layer, layer);
  }

  OperatorDef *AddFilter(const std::string &prefix, const std::string &input,
                         int in_size, int out_size, int stride, float alpha,
                         bool train) {
    auto output = "comp_" + prefix;
    std::string layer = input;
    layer = AddConvOps(layer, output, "1", in_size * alpha, in_size * alpha,
                       stride, 1, 3, true, train)
                ->output(0);
    predict.AddReluOp(layer, layer);
    layer = AddConvOps(layer, output, "2", in_size * alpha, out_size * alpha, 1,
                       0, 1, false, train)
                ->output(0);
    return predict.AddReluOp(layer, layer);
  }

  OperatorDef *AddTrain(const std::string &input) {
    std::string layer = input;
    layer = predict.AddSoftmaxOp(layer, "softmax")->output(0);
    layer = predict.AddLabelCrossEntropyOp(layer, "label", "xent")->output(0);
    layer = predict.AddAveragedLossOp(layer, "loss")->output(0);
    predict.AddAccuracyOp("softmax", "label", "top-1");
    return predict.AddAccuracyOp("softmax", "label", "top-5", 5);
  }

  OperatorDef *AddEnd(const std::string &prefix, const std::string &input,
                      int in_size, int out_size, int stride, float alpha) {
    std::string layer = input;
    layer =
        predict.AddAveragePoolOp(layer, "final_avg", stride, 0, 7)->output(0);
    return AddFcOps(layer, "last_out", in_size * alpha, out_size);
  }

  void Add(float alpha, int out_size, bool train = false) {
    predict.SetName("MobileNet");
    auto input = "data";
    auto n = 0;

    std::string layer = input;
    predict.AddInput(layer);

    layer = AddFirst("1", layer, 32, 2, alpha, train)->output(0);
    layer = AddFilter(tos2(n++), layer, 32, 64, 1, alpha, train)->output(0);
    layer = AddFilter(tos2(n++), layer, 64, 128, 2, alpha, train)->output(0);
    layer = AddFilter(tos2(n++), layer, 128, 128, 1, alpha, train)->output(0);
    layer = AddFilter(tos2(n++), layer, 128, 256, 2, alpha, train)->output(0);
    layer = AddFilter(tos2(n++), layer, 256, 256, 1, alpha, train)->output(0);
    layer = AddFilter(tos2(n++), layer, 256, 512, 2, alpha, train)->output(0);
    for (auto i = 0; i < 5; i++) {  // 6 - 10
      layer = AddFilter(tos2(n++), layer, 512, 512, 1, alpha, train)->output(0);
    }
    layer = AddFilter(tos2(n++), layer, 512, 1024, 2, alpha, train)->output(0);
    layer = AddFilter(tos2(n++), layer, 1024, 1024, 1, alpha, train)->output(0);
    layer = AddEnd("", layer, 1024, out_size, 1, alpha)->output(0);

    if (train) {
      layer = AddTrain(layer)->output(0);
    } else {
      layer = predict.AddSoftmaxOp(layer, "prob")->output(0);
    }
    predict.AddOutput(layer);
    init.AddConstantFillOp({1}, input);
  }
};

}  // namespace caffe2

#endif  // ZOO_MOBILENET_H
