#ifndef ZOO_RESNET_H
#define ZOO_RESNET_H

#include "caffe2/util/model.h"

namespace caffe2 {

class ResNetModel : public ModelUtil {
 public:
  ResNetModel(NetDef &initnet, NetDef &predictnet)
      : ModelUtil(initnet, predictnet) {}

  OperatorDef *AddConvOps(const std::string &input, const std::string &output,
                          int in_size, int out_size, int stride, int padding,
                          int kernel, bool affine = false) {
    init.AddXavierFillOp({out_size, in_size, kernel, kernel}, output + "_w");
    predict.AddInput(output + "_w");
    if (affine) {
      init.AddConstantFillOp({out_size}, output + "_b");
      predict.AddInput(output + "_b");
    }
    predict.AddConvOp(input, output + "_w", affine ? output + "_b" : "", output,
                      stride, padding, kernel);
    init.AddXavierFillOp({out_size}, output + "_scale");
    predict.AddInput(output + "_scale");
    init.AddConstantFillOp({out_size}, output + "_bias");
    predict.AddInput(output + "_bias");
    init.AddXavierFillOp({out_size}, output + "_mean");
    predict.AddInput(output + "_mean");
    init.AddXavierFillOp({out_size}, output + "_var");
    predict.AddInput(output + "_var");
    predict.AddSpatialBNOp({output, output + "_scale", output + "_bias",
                            output + "_mean", output + "_var"},
                           output + "_unique");
    init.AddXavierFillOp({out_size}, output + "_w_second");
    predict.AddInput(output + "_w_second");
    predict.AddMulOp({output + "_unique", output + "_w_second"},
                     output + "_internal");
    init.AddConstantFillOp({out_size},
                           output + "_b" + (affine ? "_second" : ""));
    predict.AddInput(output + "_b" + (affine ? "_second" : ""));
    return predict.AddAddOp({output + "_internal", output + "_b"}, output);
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
                        int in_size, int out_size, bool affine) {
    auto output = "conv" + prefix;
    std::string layer = input;
    layer = AddConvOps(layer, output, in_size, out_size, 2, 3, 7, affine)
                ->output(0);
    layer = predict.AddReluOp(output, output)->output(0);
    return predict.AddMaxPoolOp(layer, "pool" + prefix, 2, 0, 3);
  }

  OperatorDef *AddRes(const std::string &prefix, const std::string &input,
                      int in_size, int out_size, int stride = 1) {
    auto output = "res" + prefix;
    std::string layer = input;
    std::string l = input;
    if (in_size != out_size) {
      l = AddConvOps(layer, output + "_branch1", in_size, out_size, stride, 0,
                     1)
              ->output(0);
    }
    layer = AddConvOps(layer, output + "_branch2a", in_size, out_size / 4,
                       stride, 0, 1)
                ->output(0);
    predict.AddReluOp(layer, layer);
    layer = AddConvOps(layer, output + "_branch2b", out_size / 4, out_size / 4,
                       1, 1, 3)
                ->output(0);
    predict.AddReluOp(layer, layer);
    layer =
        AddConvOps(layer, output + "_branch2c", out_size / 4, out_size, 1, 0, 1)
            ->output(0);
    predict.AddSumOp({l, layer}, output);
    return predict.AddReluOp(output, output);
  }

  OperatorDef *AddTrain(const std::string &input) {
    std::string layer = input;
    layer = predict.AddSoftmaxOp(layer, "softmax")->output(0);
    layer = predict.AddLabelCrossEntropyOp(layer, "label", "xent")->output(0);
    layer = predict.AddAveragedLossOp(layer, "loss")->output(0);
    predict.AddAccuracyOp("classifier", "label", "top-1");
    return predict.AddAccuracyOp("classifier", "label", "top-5", 5);
  }

  OperatorDef *AddEnd(const std::string &prefix, const std::string &input,
                      int in_size, int out_size) {
    std::string layer = input;
    layer =
        predict.AddAveragePoolOp(layer, "pool" + prefix, 1, 0, 7)->output(0);
    return AddFcOps(layer, "fc1000", in_size, out_size);
  }

  void Add(int type, int out_size = 1000, bool train = false) {
    predict.SetName("ResNet" + std::to_string(type));
    auto input = "data";

    std::string layer = input;
    predict.AddInput(layer);

    layer = AddFirst("1", layer, 3, 64, type == 50)->output(0);
    layer = AddRes("2a", layer, 64, 256)->output(0);
    layer = AddRes("2b", layer, 256, 256)->output(0);
    layer = AddRes("2c", layer, 256, 256)->output(0);
    layer = AddRes("3a", layer, 256, 512, 2)->output(0);
    if (type == 50) {
      layer = AddRes("3b", layer, 512, 512)->output(0);
      layer = AddRes("3c", layer, 512, 512)->output(0);
      layer = AddRes("3d", layer, 512, 512)->output(0);
    } else {
      for (int i = 1; i <= (type - 60) / 12; i++) {
        layer = AddRes("3b" + std::to_string(i), layer, 512, 512)->output(0);
      }
    }
    layer = AddRes("4a", layer, 512, 1024, 2)->output(0);
    if (type == 50) {
      layer = AddRes("4b", layer, 1024, 1024)->output(0);
      layer = AddRes("4c", layer, 1024, 1024)->output(0);
      layer = AddRes("4d", layer, 1024, 1024)->output(0);
      layer = AddRes("4e", layer, 1024, 1024)->output(0);
      layer = AddRes("4f", layer, 1024, 1024)->output(0);
    } else {
      for (int i = 1; i <= (type - 10) / 4; i++) {
        layer = AddRes("4b" + std::to_string(i), layer, 1024, 1024)->output(0);
      }
    }
    layer = AddRes("5a", layer, 1024, 2048, 2)->output(0);
    layer = AddRes("5b", layer, 2048, 2048)->output(0);
    layer = AddRes("5c", layer, 2048, 2048)->output(0);
    layer = AddEnd("5", layer, 2048, out_size)->output(0);

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
