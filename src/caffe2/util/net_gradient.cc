#include "caffe2/util/net.h"

namespace caffe2 {

const std::set<std::string> trainable_ops({
    "Add",
    "AffineScale",
    "AveragedLoss",
    "AveragePool",
    "BackMean",
    "Concat",
    "Conv",
    "Diagonal",
    "Dropout",
    "EnsureCPUOutput",
    "FC",
    "LabelCrossEntropy",
    "LRN",
    "MaxPool",
    "Mul",
    "RecurrentNetwork",
    "Relu",
    "Reshape",
    "Slice",
    "Softmax",
    "SpatialBN",
    "SquaredL2",
    "SquaredL2Channel",
    "StopGradient",
    "Sum",
});

const std::set<std::string> non_trainable_ops({
    "Accuracy",
    "Cast",
    "Cout",
    "ConstantFill",
    "Iter",
    "Scale",
    "TensorProtosDBInput",
    "TimePlot",
    "ShowWorst",
});

const std::map<std::string, std::string> custom_gradient({
    {"EnsureCPUOutput", "CopyFromCPUInput"},
    {"CopyFromCPUInput", "EnsureCPUOutput"},
});

const std::set<std::string> pass_gradient({"Sum"});

const std::set<std::string> filler_ops({
    "UniformFill",
    "UniformIntFill",
    "UniqueUniformFill",
    "ConstantFill",
    "GaussianFill",
    "XavierFill",
    "MSRAFill",
    "RangeFill",
    "LengthsRangeFill",
});

const std::string gradient_suffix("_grad");

// Gradient

void NetUtil::AddGradientOps() { AddGradientOps(*this); }

void NetUtil::AddGradientOps(NetUtil& target) const {
  std::map<std::string, std::pair<int, int>> split_inputs;
  std::map<std::string, std::string> pass_replace;
  std::set<std::string> stop_inputs;
  auto ops = CollectGradientOps(split_inputs);
  for (auto op : ops) {
    target.AddGradientOps(op, split_inputs, pass_replace, stop_inputs);
  }
}

bool net_util_op_has_output(const OperatorDef& op,
                            const std::set<std::string>& names) {
  for (const auto& output : op.output()) {
    if (names.find(output) != names.end()) {
      return true;
    }
  }
  return false;
}

OperatorDef* NetUtil::AddGradientOp(OperatorDef& op) {
  OperatorDef* grad = NULL;
  vector<GradientWrapper> output(op.output_size());
  for (auto i = 0; i < output.size(); i++) {
    output[i].dense_ = op.output(i) + gradient_suffix;
  }
  GradientOpsMeta meta = GetGradientForOp(op, output);
  if (meta.ops_.size()) {
    for (auto& m : meta.ops_) {
      auto op = net.add_op();
      op->CopyFrom(m);
      if (grad == NULL) {
        grad = op;
      }
    }
  }
  return grad;
}

OperatorDef* NetUtil::AddGradientOps(
    OperatorDef& op, std::map<std::string, std::pair<int, int>>& split_inputs,
    std::map<std::string, std::string>& pass_replace,
    std::set<std::string>& stop_inputs) {
  OperatorDef* grad = NULL;
  if (custom_gradient.find(op.type()) != custom_gradient.end()) {
    grad = net.add_op();
    grad->set_type(custom_gradient.at(op.type()));
    for (auto arg : op.arg()) {
      auto copy = grad->add_arg();
      copy->CopyFrom(arg);
    }
    for (auto output : op.output()) {
      grad->add_input(output + gradient_suffix);
    }
    for (auto input : op.input()) {
      grad->add_output(input + gradient_suffix);
    }
  } else if (pass_gradient.find(op.type()) != pass_gradient.end()) {
    for (auto input : op.input()) {
      auto in = input + gradient_suffix;
      if (split_inputs.count(in) && split_inputs[in].first > 0) {
        split_inputs[in].first--;
        in += "_sum_" + std::to_string(split_inputs[in].first);
      }
      pass_replace[in] = op.output(0) + gradient_suffix;
    }
  } else if (op.type() == "StopGradient" ||
             net_util_op_has_output(op, stop_inputs)) {
    for (const auto& input : op.input()) {
      stop_inputs.insert(input);
    }
  } else {
    grad = AddGradientOp(op);
    if (grad == NULL) {
      std::cerr << "no gradient for operator " << op.type() << std::endl;
    }
  }
  if (grad != NULL) {
    grad->set_is_gradient_op(true);
    for (auto i = 0; i < grad->output_size(); i++) {
      auto output = grad->output(i);
      if (split_inputs.count(output) && split_inputs[output].first > 0) {
        split_inputs[output].first--;
        grad->set_output(
            i, output + "_sum_" + std::to_string(split_inputs[output].first));
      }
    }
    for (auto i = 0; i < grad->input_size(); i++) {
      auto input = grad->input(i);
      if (pass_replace.count(input)) {
        grad->set_input(i, pass_replace[input]);
        pass_replace.erase(input);
      }
    }
    // fix for non-in-place SpatialBN
    if (grad->type() == "SpatialBNGradient" &&
        grad->input(2) == grad->output(0)) {
      pass_replace[grad->output(0)] = grad->output(0) + "_fix";
      grad->set_output(0, grad->output(0) + "_fix");
    }
  }
  // merge split gradients with sum
  for (auto& p : split_inputs) {
    if (p.second.first == 0) {
      std::vector<std::string> inputs;
      for (int i = 0; i < p.second.second; i++) {
        auto input = p.first + "_sum_" + std::to_string(i);
        if (pass_replace.count(input)) {
          auto in = pass_replace[input];
          pass_replace.erase(input);
          input = in;
        }
        inputs.push_back(input);
      }
      AddSumOp(inputs, p.first);
      p.second.first--;
    }
  }
  return grad;
}

std::vector<OperatorDef> NetUtil::CollectGradientOps(
    std::map<std::string, std::pair<int, int>>& split_inputs) const {
  std::set<std::string> external_inputs(net.external_input().begin(),
                                        net.external_input().end());
  std::vector<OperatorDef> gradient_ops;
  std::map<std::string, int> input_count;
  for (auto& op : net.op()) {
    if (trainable_ops.find(op.type()) != trainable_ops.end()) {
      gradient_ops.push_back(op);
      for (auto& input : op.input()) {
        auto& output = op.output();
        if (std::find(output.begin(), output.end(), input) == output.end()) {
          input_count[input]++;
          if (input_count[input] > 1) {
            split_inputs[input + gradient_suffix] = {input_count[input],
                                                     input_count[input]};
          }
        }
      }
    } else if (non_trainable_ops.find(op.type()) == non_trainable_ops.end()) {
      CAFFE_THROW("unknown backprop operator type: " + op.type());
    }
  }
  std::reverse(gradient_ops.begin(), gradient_ops.end());
  return gradient_ops;
}

// Collectors

std::map<std::string, int> NetUtil::CollectParamSizes() {
  std::map<std::string, int> sizes;
  for (const auto& op : net.op()) {
    if (filler_ops.find(op.type()) != filler_ops.end()) {
      for (const auto& arg : op.arg()) {
        if (arg.name() == "shape") {
          auto size = 1;
          for (auto i : arg.ints()) {
            size *= i;
          }
          sizes[op.output(0)] = size;
        }
      }
    }
  }
  return sizes;
}

std::vector<std::string> NetUtil::CollectParams() {
  std::vector<std::string> params;
  std::set<std::string> external_inputs(net.external_input().begin(),
                                        net.external_input().end());
  for (const auto& op : net.op()) {
    auto& output = op.output();
    if (trainable_ops.find(op.type()) != trainable_ops.end()) {
      for (const auto& input : op.input()) {
        if (external_inputs.find(input) != external_inputs.end()) {
          if (std::find(output.begin(), output.end(), input) == output.end()) {
            params.push_back(input);
          }
        }
      }
    }
  }
  return params;
}

std::set<std::string> NetUtil::CollectLayers(const std::string& layer,
                                             bool forward) {
  std::map<std::string, std::set<std::string>> lookup;
  for (auto& op : net.op()) {
    for (auto& input : op.input()) {
      for (auto& output : op.output()) {
        lookup[forward ? input : output].insert(forward ? output : input);
      }
    }
  }
  std::set<std::string> result;
  for (std::set<std::string> step({layer}); step.size();) {
    std::set<std::string> next;
    for (auto& l : step) {
      if (result.find(l) == result.end()) {
        result.insert(l);
        for (auto& n : lookup[l]) {
          next.insert(n);
        }
      }
    }
    step = next;
  }
  return result;
}

void NetUtil::CheckLayerAvailable(const std::string& layer) {
  std::vector<std::pair<std::string, std::string>> available_layers(
      {{Input(0), "Input"}});
  auto layer_found = (Input(0) == layer);
  for (const auto& op : net.op()) {
    if (op.input(0) != op.output(0)) {
      available_layers.push_back({op.output(0), op.type()});
      layer_found |= (op.output(0) == layer);
    }
  }
  if (!layer_found) {
    std::cerr << "available layers:" << std::endl;
    for (auto& layer : available_layers) {
      std::cerr << "  " << layer.first << " (" << layer.second << ")"
                << std::endl;
    }
    LOG(FATAL) << "~ no layer with name " << layer << " in model.";
  }
}

}  // namespace caffe2
