#include "caffe2/util/net.h"

namespace caffe2 {

const std::set<std::string> trainable_ops({
    "Add",
    "AffineScale",
    "AveragedLoss",
    "AveragePool",
    "BackMean",
	"Cast",
    "Concat",
    "Conv",
	"ConvTranspose",
	"Copy",
    "Diagonal",
    "Dropout",
    "EnsureCPUOutput",
    "FC",
    "LabelCrossEntropy",
    "LRN",
    "MaxPool",
    "Mul",
	"Pow",
    "RecurrentNetwork",
	"ReduceBackMean",
	"ReduceBackSum",
	"ReduceTailSum",
    "Relu",
	"LeakyRelu",
	"L1Distance",
    "Reshape",
	"Scale", 
	"Sigmoid",
    "Slice",
    "Softmax",
	"SoftmaxWithLoss",
    "SpatialBN",
    "SquaredL2",
    "SquaredL2Channel",
	"SquaredL2Distance",
	"Sub",
	"StopGradient",
    "Sum",
	"Tanh",
	"Transpose",
});

const std::set<std::string> non_trainable_ops({
    "Accuracy",  
	"Cout", 
	"ConstantFill", 
	"GaussianFill", 
	"Iter", 
    "TensorProtosDBInput", 
	"TimePlot", 
	"ShowWorst",
	"Save",
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

void NetUtil::AddGradientOps() {
  std::map<std::string, std::pair<int, int>> split_inputs;
  std::map<std::string, std::string> pass_replace;
  std::set<std::string> stop_inputs;
  for (auto op : CollectGradientOps(split_inputs)) {
    AddGradientOp(op, split_inputs, pass_replace, stop_inputs);
  }
}

void NetUtil::AddGradientOps(NetUtil & net2) {
  std::map<std::string, std::pair<int, int>> split_inputs;
  std::map<std::string, std::string> pass_replace;
  std::set<std::string> stop_inputs;
  for (auto op : CollectGradientOps(split_inputs)) {
    net2.AddGradientOp(op, split_inputs, pass_replace, stop_inputs);
  }
  //将所有当前网络自己生成的输出保存到一个set
  std::set<std::string> outputs;
  for (auto op : net2.net.op())
	  for(auto output : op.output())
		  outputs.insert(output);
  //所有不是当前网络输出的blob，如果作为输入就添加external_input
  for (auto op : net2.net.op())
	  for(auto input : op.input())
		  if(outputs.end() == outputs.find(input))
			  net2.AddInput(input);
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

OperatorDef* NetUtil::AddGradientOp(
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
      pass_replace[input + gradient_suffix] = op.output(0) + gradient_suffix;
    }
  } else if (op.type() == "StopGradient" ||
             net_util_op_has_output(op, stop_inputs)) {
    for (const auto& input : op.input()) {
      stop_inputs.insert(input);
    }
  } else {
    grad = net.add_op();
    vector<GradientWrapper> output(op.output_size());
    for (auto i = 0; i < output.size(); i++) {
      output[i].dense_ = op.output(i) + gradient_suffix;
    }
    GradientOpsMeta meta = GetGradientForOp(op, output);
    if (meta.ops_.size()) {
      if (meta.ops_.size() > 1) {
        std::cerr << "multiple gradients for operator (" << op.type();
        for (auto& o : meta.ops_) {
          std::cerr << " " << o.type();
        }
        std::cerr << ")" << std::endl;
      }
      grad->CopyFrom(meta.ops_[0]);
    } else {
      std::cerr << "no gradient for operator " << op.type() << std::endl;
    }
  }
  if (grad != NULL) {
    grad->set_is_gradient_op(true);
    for (auto i = 0; i < grad->output_size(); i++) {
      auto output = grad->output(i);
      if (split_inputs.count(output)) {
        split_inputs[output].first--;
        grad->set_output(
            i, output + "_sum_" + std::to_string(split_inputs[output].first));
        if (split_inputs[output].first == 0) {
          std::vector<std::string> inputs;
          for (int i = 0; i < split_inputs[output].second; i++) {
            inputs.push_back(output + "_sum_" + std::to_string(i));
          }
          AddSumOp(inputs, output);
        }
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
  return grad;
}

std::vector<OperatorDef> NetUtil::CollectGradientOps(
    std::map<std::string, std::pair<int, int>>& split_inputs) {
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
            split_inputs[input + "_grad"] = {input_count[input],
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
      {{net.external_input(0), "Input"}});
  auto layer_found = (net.external_input(0) == layer);
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
