#ifndef ZOO_RESNET_H
#define ZOO_RESNET_H

#include "caffe2/util/model.h"

namespace caffe2 {

std::string tos(int i) { return std::to_string(i); }

class ResNetModel : public ModelUtil {
 public:
  ResNetModel(NetDef &initnet, NetDef &predictnet)
      : ModelUtil(initnet, predictnet) {}

  OperatorDef *AddConvOps(const std::string &input, const std::string &output,
                          const std::string &infix, int in_size, int out_size,
                          int stride, int padding, int kernel, bool train) {
    auto b = output + (infix.size() ? "_conv_" + infix : "");
    init.AddMSRAFillOp({out_size, in_size, kernel, kernel}, b + "_w");
    predict.AddInput(b + "_w");
    predict.AddConvOp(input, b + "_w", "", b, stride, padding, kernel);
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
        0.001, 0.1, !train);
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
                        int out_size, int stride, bool train) {
    auto output = "conv" + prefix;
    std::string layer = input;
    layer = AddConvOps(layer, output, "relu", 3, out_size, stride, 3, 7, train)
                ->output(0);
    layer = predict.AddReluOp(layer, layer)->output(0);
    return predict.AddMaxPoolOp(layer, "pool" + prefix, 2, 0, 3);
  }

  OperatorDef *AddRes2(const std::string &prefix, const std::string &input,
                       int in_size, int out_size, int stride, bool train) {
    auto output = "comp_" + prefix;
    std::string layer = input;
    layer = AddConvOps(layer, output, "1", in_size, out_size, 1, 1, 3, train)
                ->output(0);
    predict.AddReluOp(layer, layer);
    layer =
        AddConvOps(layer, output, "2", out_size, out_size, stride, 1, 3, train)
            ->output(0);
    auto in = input, out = output + "_sum_3";
    if (in_size != out_size) {
      in = AddConvOps(input, "shortcut_projection_" + prefix, "", in_size,
                      out_size, stride, 0, 1, train)
               ->output(0);
    }
    predict.AddSumOp({in, layer}, out);
    return predict.AddReluOp(out, out);
  }

  OperatorDef *AddRes3(const std::string &prefix, const std::string &input,
                       int in_size, int out_size, int stride, bool train) {
    auto output = "comp_" + prefix;
    std::string layer = input;
    layer =
        AddConvOps(layer, output, "1", in_size, out_size / 4, 1, 0, 1, train)
            ->output(0);
    predict.AddReluOp(layer, layer);
    layer = AddConvOps(layer, output, "2", out_size / 4, out_size / 4, stride,
                       1, 3, train)
                ->output(0);
    predict.AddReluOp(layer, layer);
    layer =
        AddConvOps(layer, output, "3", out_size / 4, out_size, 1, 0, 1, train)
            ->output(0);
    auto in = input, out = output + "_sum_3";
    if (in_size != out_size) {
      in = AddConvOps(input, "shortcut_projection_" + prefix, "", in_size,
                      out_size, stride, 0, 1, train)
               ->output(0);
    }
    predict.AddSumOp({in, layer}, out);
    return predict.AddReluOp(out, out);
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
                      int in_size, int out_size, int stride) {
    std::string layer = input;
    layer =
        predict.AddAveragePoolOp(layer, "final_avg", stride, 0, 7)->output(0);
    return AddFcOps(layer, "last_out_L1000", in_size, out_size);
  }

  OperatorDef *AddBlock2(int &n, const std::string &layer, int in_size,
                         int out_size, int stride, int depth, bool train) {
    auto op = AddRes2(tos(n++), layer, in_size, out_size, stride, train);
    for (int i = 1; i < depth; i++) {
      op = AddRes2(tos(n++), op->output(0), out_size, out_size, 1, train);
    }
    return op;
  }

  OperatorDef *AddBlock3(int &n, const std::string &layer, int in_size,
                         int out_size, int stride, int depth, bool train) {
    auto op = AddRes3(tos(n++), layer, in_size, out_size, stride, train);
    for (int i = 1; i < depth; i++) {
      op = AddRes3(tos(n++), op->output(0), out_size, out_size, 1, train);
    }
    return op;
  }

  void Add(int type, int out_size, bool train = false) {
    predict.SetName("ResNet" + std::to_string(type));
    auto input = "data";
    auto n = 0;

    std::string layer = input;
    predict.AddInput(layer);

    // <silly>
    auto depth_2 = std::min(3, type / 16 + 1);
    auto depth_3 = 1 << ((type + 170) / 100);
    auto depth_4 = (type - 2) / (type < 50 ? 2 : 3) - (2 * depth_2 + depth_3);
    auto depth_5 = depth_2;
    // </silly>

    if (type < 50) {
      layer = AddFirst("1", layer, 64, 2, train)->output(0);
      layer = AddBlock2(n, layer, 64, 64, 1, depth_2, train)->output(0);
      layer = AddBlock2(n, layer, 64, 128, 2, depth_3, train)->output(0);
      layer = AddBlock2(n, layer, 128, 256, 2, depth_4, train)->output(0);
      layer = AddBlock2(n, layer, 256, 512, 2, depth_5, train)->output(0);
      layer = AddEnd(tos(n++), layer, 512, out_size, 1)->output(0);
    } else {
      layer = AddFirst("1", layer, 64, 2, train)->output(0);
      layer = AddBlock3(n, layer, 64, 256, 1, depth_2, train)->output(0);
      layer = AddBlock3(n, layer, 256, 512, 2, depth_3, train)->output(0);
      layer = AddBlock3(n, layer, 512, 1024, 2, depth_4, train)->output(0);
      layer = AddBlock3(n, layer, 1024, 2048, 2, depth_5, train)->output(0);
      layer = AddEnd(tos(n++), layer, 2048, out_size, 1)->output(0);
    }

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

#endif  // ZOO_RESNET_H
