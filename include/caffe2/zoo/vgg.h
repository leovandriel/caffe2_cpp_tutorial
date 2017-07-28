#ifndef ZOO_VGG_H
#define ZOO_VGG_H

#include "caffe2/util/model.h"

namespace caffe2 {

class VGGModel : public ModelUtil {
 public:
  VGGModel(NetDef &init_net, NetDef &predict_net)
      : ModelUtil(init_net, predict_net) {}

  OperatorDef *AddConvOps(const std::string &input, const std::string &output,
                          int in_size, int out_size, int stride, int padding,
                          int kernel) {
    init_.AddXavierFillOp({out_size, in_size, kernel, kernel}, output + "_w");
    predict_.AddInput(output + "_w");
    init_.AddConstantFillOp({out_size}, output + "_b");
    predict_.AddInput(output + "_b");
    predict_.AddConvOp(input, output + "_w", output + "_b", output, stride,
                       padding, kernel);
    return predict_.AddReluOp(output, output);
  }

  OperatorDef *AddFcOps(const std::string &input, const std::string &output,
                        int in_size, int out_size, bool relu = false,
                        float dropout = 0.5) {
    init_.AddXavierFillOp({out_size, in_size}, output + "_w");
    predict_.AddInput(output + "_w");
    init_.AddConstantFillOp({out_size}, output + "_b");
    predict_.AddInput(output + "_b");
    auto op = predict_.AddFcOp(input, output + "_w", output + "_b", output);
    if (!relu) return op;
    predict_.AddReluOp(output, output);
    return predict_.AddDropoutOp(output, output, dropout);
  }

  OperatorDef *AddTrain(const std::string &input) {
    std::string layer = input;
    layer = predict_.AddSoftmaxOp(layer, "softmax")->output(0);
    layer = predict_.AddLabelCrossEntropyOp(layer, "label", "xent")->output(0);
    layer = predict_.AddAveragedLossOp(layer, "loss")->output(0);
    predict_.AddAccuracyOp("classifier", "label", "top-1");
    return predict_.AddAccuracyOp("classifier", "label", "top-5", 5);
  }

  void Add(int type, int out_size = 1000, bool train = false) {
    predict_.SetName("VGG" + std::to_string(type));
    auto input = "data";
  std:
    string layer = input;
    predict_.AddInput(layer);

    layer = AddConvOps(layer, "conv1_1", 3, 64, 1, 1, 3)->output(0);
    layer = AddConvOps(layer, "conv1_2", 64, 64, 1, 1, 3)->output(0);
    layer = predict_.AddMaxPoolOp(layer, "pool1", 2, 0, 2)->output(0);
    layer = AddConvOps(layer, "conv2_1", 64, 128, 1, 1, 3)->output(0);
    layer = AddConvOps(layer, "conv2_2", 128, 128, 1, 1, 3)->output(0);
    layer = predict_.AddMaxPoolOp(layer, "pool2", 2, 0, 2)->output(0);
    layer = AddConvOps(layer, "conv3_1", 128, 256, 1, 1, 3)->output(0);
    layer = AddConvOps(layer, "conv3_2", 256, 256, 1, 1, 3)->output(0);
    layer = AddConvOps(layer, "conv3_3", 256, 256, 1, 1, 3)->output(0);
    if (type == 19) {
      layer = AddConvOps(layer, "conv3_4", 256, 256, 1, 1, 3)->output(0);
    }
    layer = predict_.AddMaxPoolOp(layer, "pool3", 2, 0, 2)->output(0);
    layer = AddConvOps(layer, "conv4_1", 256, 512, 1, 1, 3)->output(0);
    layer = AddConvOps(layer, "conv4_2", 512, 512, 1, 1, 3)->output(0);
    layer = AddConvOps(layer, "conv4_3", 512, 512, 1, 1, 3)->output(0);
    if (type == 19) {
      layer = AddConvOps(layer, "conv4_4", 512, 512, 1, 1, 3)->output(0);
    }
    layer = predict_.AddMaxPoolOp(layer, "pool4", 2, 0, 2)->output(0);
    layer = AddConvOps(layer, "conv5_1", 512, 512, 1, 1, 3)->output(0);
    layer = AddConvOps(layer, "conv5_2", 512, 512, 1, 1, 3)->output(0);
    layer = AddConvOps(layer, "conv5_3", 512, 512, 1, 1, 3)->output(0);
    if (type == 19) {
      layer = AddConvOps(layer, "conv5_4", 512, 512, 1, 1, 3)->output(0);
    }
    layer = predict_.AddMaxPoolOp(layer, "pool5", 2, 0, 2)->output(0);
    layer = AddFcOps(layer, "fc6", 25088, 4096, true)->output(0);
    layer = AddFcOps(layer, "fc7", 4096, 4096, true)->output(0);
    layer = AddFcOps(layer, "fc8", 4096, out_size)->output(0);

    if (train) {
      layer = AddTrain(layer)->output(0);
    } else {
      layer = predict_.AddSoftmaxOp(layer, "prob")->output(0);
    }
    predict_.AddOutput(layer);
    init_.AddConstantFillOp({1}, input);
  }
};

}  // namespace caffe2

#endif  // ZOO_VGG_H
