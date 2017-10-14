#ifndef ZOO_GOOGLENET_H
#define ZOO_GOOGLENET_H

#include "caffe2/util/model.h"

namespace caffe2 {

class GoogleNetModel : public ModelUtil {
 public:
  GoogleNetModel(NetDef &initnet, NetDef &predictnet)
      : ModelUtil(initnet, predictnet) {}

  OperatorDef *AddConvOps(const std::string &input, const std::string &output,
                          int in_size, int out_size, int stride, int padding,
                          int kernel) {
    init.AddXavierFillOp({out_size, in_size, kernel, kernel}, output + "_w");
    predict.AddInput(output + "_w");
    init.AddConstantFillOp({out_size}, output + "_b");
    predict.AddInput(output + "_b");
    predict.AddConvOp(input, output + "_w", output + "_b", output, stride,
                      padding, kernel);
    return predict.AddReluOp(output, output);
  }

  OperatorDef *AddFcOps(const std::string &input, const std::string &output,
                        int in_size, int out_size, bool relu,
                        float dropout = 0.5) {
    init.AddXavierFillOp({out_size, in_size}, output + "_w");
    predict.AddInput(output + "_w");
    init.AddConstantFillOp({out_size}, output + "_b");
    predict.AddInput(output + "_b");
    auto op = predict.AddFcOp(input, output + "_w", output + "_b", output);
    if (!relu) return op;
    predict.AddReluOp(output, output);
    return predict.AddDropoutOp(output, output, dropout);
  }

  OperatorDef *AddFirst(const std::string &prefix, const std::string &input,
                        int in_size, int out_size) {
    auto output = "conv" + prefix + "/";
    std::string layer = input;
    layer = AddConvOps(layer, output + "7x7_s2", in_size, out_size, 2, 3, 7)
                ->output(0);
    layer = predict.AddMaxPoolOp(layer, "pool" + prefix + "/3x3_s2", 2, 0, 3)
                ->output(0);
    return predict.AddLrnOp(layer, "pool1/norm1", 5, 0.0001, 0.75, 1);
  }

  OperatorDef *AddSecond(const std::string &prefix, const std::string &input,
                         int in_size, int out_size) {
    auto output = "conv" + prefix + "/3x3";
    std::string layer = input;
    layer =
        AddConvOps(layer, output + "_reduce", in_size, out_size / 3, 1, 0, 1)
            ->output(0);
    layer = AddConvOps(layer, output, in_size, out_size, 1, 1, 3)->output(0);
    return predict.AddLrnOp(layer, "conv2/norm2", 5, 0.0001, 0.75, 1);
  }

  OperatorDef *AddPool(const std::string &prefix, const std::string &input) {
    return predict.AddMaxPoolOp(input, "pool" + prefix + "/3x3_s2", 2, 0, 3);
  }

  OperatorDef *AddInception(const std::string &prefix, const std::string &input,
                            std::vector<int> sizes) {
    auto output = "inception_" + prefix + "/";
    std::string layer = input;
    std::vector<std::string> layers;
    for (int i = 0, kernel = 1; i < 3; i++, kernel += 2) {
      auto b = output + std::to_string(kernel) + "x" + std::to_string(kernel);
      if (i) {
        layer = AddConvOps(input, b + "_reduce", sizes[0], sizes[kernel - 1], 1,
                           0, 1)
                    ->output(0);
      }
      layers.push_back(
          AddConvOps(layer, b, sizes[kernel - 1], sizes[kernel], 1, i, kernel)
              ->output(0));
    }
    layer = predict.AddMaxPoolOp(input, output + "pool", 1, 1, 3)->output(0);
    layers.push_back(
        AddConvOps(layer, layer + "_proj", sizes[0], sizes[6], 1, 0, 1)
            ->output(0));
    return predict.AddConcatOp(layers, output + "output");
  }

  OperatorDef *AddSide(const std::string &prefix, const std::string &input,
                       std::vector<int> sizes, int out_size) {
    auto output = "loss" + prefix + "/";
    std::string layer = input;
    layer = predict.AddAveragePoolOp(layer, output + "ave_pool", 3, 0, 5)
                ->output(0);
    layer = AddConvOps(layer, output + "conv", sizes[0], sizes[1], 1, 0, 1)
                ->output(0);
    layer = AddFcOps(layer, output + "fc", sizes[2], sizes[3], true, 0.7)
                ->output(0);
    return AddFcOps(layer, output + "classifier", sizes[3], out_size, false);
  }

  OperatorDef *AddTrain(const std::string &prefix, const std::string &input) {
    auto output = "loss" + prefix + "/";
    std::string layer = input;
    layer = predict.AddSoftmaxOp(layer, output + "softmax")->output(0);
    layer = predict.AddLabelCrossEntropyOp(layer, "label", output + "xent")
                ->output(0);
    layer =
        predict.AddAveragedLossOp(layer, output + "loss" + prefix)->output(0);
    predict.AddAccuracyOp(output + "classifier", "label", output + "top-1");
    return predict.AddAccuracyOp(output + "classifier", "label",
                                 output + "top-5", 5);
  }

  OperatorDef *AddEnd(const std::string &prefix, const std::string &input,
                      int in_size, int out_size) {
    auto output = "loss" + prefix + "/";
    std::string layer = input;
    layer = predict.AddAveragePoolOp(layer, "pool5/7x7_s1", 1, 0, 7)->output(0);
    layer = predict.AddDropoutOp(layer, layer, 0.4)->output(0);
    return AddFcOps(layer, output + "classifier", in_size, out_size, false);
  }

  void Add(int out_size, bool train = false) {
    predict.SetName("GoogleNet");
    auto input = "data";

    std::string layer = input;
    predict.AddInput(layer);
    layer = AddFirst("1", layer, 3, 64)->output(0);
    layer = AddSecond("2", layer, 64, 192)->output(0);
    layer = AddPool("2", layer)->output(0);
    layer =
        AddInception("3a", layer, {192, 64, 96, 128, 16, 32, 32})->output(0);
    layer =
        AddInception("3b", layer, {256, 128, 128, 192, 32, 96, 64})->output(0);
    layer = AddPool("3", layer)->output(0);
    layer =
        AddInception("4a", layer, {480, 192, 96, 208, 16, 48, 64})->output(0);
    if (train) {
      auto last = layer;
      layer = AddSide("1", layer, {512, 128, 2048, 1024}, out_size)->output(0);
      layer = AddTrain("1", layer)->output(0);
      layer = last;
    }
    layer =
        AddInception("4b", layer, {512, 160, 112, 224, 24, 64, 64})->output(0);
    layer =
        AddInception("4c", layer, {512, 128, 128, 256, 24, 64, 64})->output(0);
    layer =
        AddInception("4d", layer, {512, 112, 144, 288, 32, 64, 64})->output(0);
    if (train) {
      auto last = layer;
      layer = AddSide("2", layer, {528, 128, 2048, 1024}, out_size)->output(0);
      layer = AddTrain("2", layer)->output(0);
      layer = last;
    }
    layer = AddInception("4e", layer, {528, 256, 160, 320, 32, 128, 128})
                ->output(0);
    layer = AddPool("4", layer)->output(0);
    layer = AddInception("5a", layer, {832, 256, 160, 320, 32, 128, 128})
                ->output(0);
    layer = AddInception("5b", layer, {832, 384, 192, 384, 48, 128, 128})
                ->output(0);
    layer = AddEnd("3", layer, 1024, out_size)->output(0);
    if (train) {
      layer = AddTrain("3", layer)->output(0);
    } else {
      layer = predict.AddSoftmaxOp(layer, "prob")->output(0);
    }
    predict.AddOutput(layer);
    init.AddConstantFillOp({1}, input);
  }
};

}  // namespace caffe2

#endif  // ZOO_GOOGLENET_H
