#ifndef ZOO_ALEXNET_H
#define ZOO_ALEXNET_H

#include "caffe2/util/model.h"

namespace caffe2 {

class AlexNetModel : public ModelUtil {
 public:
  AlexNetModel(NetDef& init_net, NetDef& predict_net):
    ModelUtil(init_net, predict_net) {}

  OperatorDef *AddConvOps(const std::string &prefix, const std::string &input, int in_size, int out_size, int stride, int padding, int kernel, bool group) {
    auto output = "conv" + prefix;
    init_.AddXavierFillOp({ out_size, in_size, kernel, kernel }, output + "_w");
    predict_.AddInput(output + "_w");
    init_.AddConstantFillOp({ out_size }, output + "_b");
    predict_.AddInput(output + "_b");
    auto conv = predict_.AddConvOp(input, output + "_w", output + "_b", output, stride, padding, kernel);
    if (group) {
      auto arg = conv->add_arg();
      arg->set_name("group");
      arg->set_i(2);
    }
    return predict_.AddReluOp(output, output);
  }

  OperatorDef *AddConvPool(const std::string &prefix, const std::string &input, int in_size, int out_size, int stride, int padding, int kernel, bool group) {
    auto output = "conv" + prefix;
    auto op = AddConvOps(prefix, input, in_size, out_size, stride, padding, kernel, group);
    predict_.AddLrnOp(output, "norm" + prefix, 5, 0.0001, 0.75, 1);
    return predict_.AddMaxPoolOp("norm" + prefix, "pool" + prefix, 2, 0, 3);
  }

  OperatorDef *AddFc(const std::string &prefix, const std::string &input, int in_size, int out_size, bool relu) {
    auto output = "fc" + prefix;
    init_.AddXavierFillOp({ out_size, in_size }, output + "_w");
    predict_.AddInput(output + "_w");
    init_.AddConstantFillOp({ out_size }, output + "_b");
    predict_.AddInput(output + "_b");
    auto op = predict_.AddFcOp(input, output + "_w", output + "_b", output);
    if (!relu) return op;
    predict_.AddReluOp(output, output);
    return predict_.AddDropoutOp(output, output, 0.5);
  }

  OperatorDef *AddTrain(const std::string &prefix, const std::string &input) {
    auto output = "loss" + prefix + "/";
    std::string layer = input;
    layer = predict_.AddSoftmaxOp(layer, output + "softmax")->output(0);
    layer = predict_.AddLabelCrossEntropyOp(layer, "label", output + "xent")->output(0);
    layer = predict_.AddAveragedLoss(layer, output + "loss" + prefix)->output(0);
    predict_.AddAccuracyOp(output + "classifier", "label", output + "top-1");
    return predict_.AddAccuracyOp(output + "classifier", "label", output + "top-5", 5);
  }

  void Add(int out_size = 1000, bool train = false) {
    predict_.SetName("AlexNet");
    auto input = "data";
    std:string layer = input;
    predict_.AddInput(layer);
    layer = AddConvPool("1", layer, 3, 96, 4, 0, 11, false)->output(0);
    layer = AddConvPool("2", layer, 48, 256, 1, 2, 5, true)->output(0);
    layer = AddConvOps("3", layer, 256, 384, 1, 1, 3, false)->output(0);
    layer = AddConvOps("4", layer, 192, 384, 1, 1, 3, true)->output(0);
    layer = AddConvOps("5", layer, 192, 256, 1, 1, 3, true)->output(0);
    layer = predict_.AddMaxPoolOp(layer, "pool5", 2, 0, 3)->output(0);
    layer = AddFc("6", layer, 9216, 4096, true)->output(0);
    layer = AddFc("7", layer, 4096, 4096, true)->output(0);
    layer = AddFc("8", layer, 4096, out_size, false)->output(0);
    if (train) {
      layer = AddTrain("1", layer)->output(0);
    } else {
      layer = predict_.AddSoftmaxOp(layer, "prob")->output(0);
    }
    predict_.AddOutput(layer);
    init_.AddConstantFillOp({ 1 }, input);
  }

};

}  // namespace caffe2

#endif  // ZOO_ALEXNET_H
