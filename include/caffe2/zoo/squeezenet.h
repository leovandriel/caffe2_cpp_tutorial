#ifndef ZOO_SQUEEZENET_H
#define ZOO_SQUEEZENET_H

#include "caffe2/util/model.h"

namespace caffe2 {

class SqueezeNetModel : public ModelUtil {
 public:
  SqueezeNetModel(NetDef &initnet, NetDef &predictnet)
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

  OperatorDef *AddFirst(const std::string &prefix, const std::string &input,
                        int in_size, int out_size) {
    return AddConvOps(input, "conv" + prefix, in_size, out_size, 2, 0, 3);
  }

  OperatorDef *AddPool(const std::string &prefix, const std::string &input) {
    return predict.AddMaxPoolOp(input, "pool" + prefix, 2, 0, 3);
  }

  OperatorDef *AddFire(const std::string &prefix, const std::string &input,
                       int in_size, int out_size) {
    auto output = "fire" + prefix + "/";
    std::string layer = input;
    layer =
        AddConvOps(layer, output + "squeeze1x1", in_size, out_size / 4, 1, 0, 1)
            ->output(0);
    auto l =
        AddConvOps(layer, output + "expand1x1", out_size / 4, out_size, 1, 0, 1)
            ->output(0);
    layer =
        AddConvOps(layer, output + "expand3x3", out_size / 4, out_size, 1, 1, 3)
            ->output(0);
    return predict.AddConcatOp({l, layer}, output + "concat");
  }

  OperatorDef *AddEnd(const std::string &prefix, const std::string &input,
                      int in_size, int out_size) {
    auto output = "loss" + prefix + "/";
    std::string layer = input;
    layer = predict.AddDropoutOp(layer, layer, 0.5)->output(0);
    layer = AddConvOps(layer, "conv" + prefix, in_size, out_size, 1, 0, 1)
                ->output(0);
    auto op = predict.AddAveragePoolOp(layer, "pool" + prefix, 1, 0, 0);
    auto arg = op->add_arg();
    arg->set_name("global_pooling");
    arg->set_i(1);
    return op;
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

  void Add(int out_size = 1000, bool train = false) {
    predict.SetName("SqueezeNet");
    auto input = "data";

    std::string layer = input;
    predict.AddInput(layer);
    layer = AddFirst("1", layer, 3, 64)->output(0);
    layer = AddPool("1", layer)->output(0);
    layer = AddFire("2", layer, 64, 64)->output(0);
    layer = AddFire("3", layer, 128, 64)->output(0);
    layer = AddPool("3", layer)->output(0);
    layer = AddFire("4", layer, 128, 128)->output(0);
    layer = AddFire("5", layer, 256, 128)->output(0);
    layer = AddPool("5", layer)->output(0);
    layer = AddFire("6", layer, 256, 192)->output(0);
    layer = AddFire("7", layer, 384, 192)->output(0);
    layer = AddFire("8", layer, 384, 256)->output(0);
    layer = AddFire("9", layer, 512, 256)->output(0);
    layer = AddEnd("10", layer, 512, out_size)->output(0);

    if (train) {
      layer = AddTrain("1", layer)->output(0);
    } else {
      layer = predict.AddSoftmaxOp(layer, "softmax")->output(0);
    }
    predict.AddOutput(layer);
    init.AddConstantFillOp({1}, input);
  }
};

}  // namespace caffe2

#endif  // ZOO_SQUEEZENET_H
