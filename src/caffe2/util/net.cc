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
  "Relu",
  "Reshape",
  "Slice",
  "Softmax",
  "SpatialBN",
  "Sum",
});

const std::set<std::string> non_trainable_ops({
  "Accuracy",
  "Cout",
  "ConstantFill",
  "TensorProtosDBInput",
});

const std::map<std::string, std::string> custom_gradient({
  { "EnsureCPUOutput", "CopyFromCPUInput" },
  { "CopyFromCPUInput", "EnsureCPUOutput" },
});

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

const std::set<std::string> non_inplace_ops({
  "Dropout", // TODO: see if "they" fixed dropout on cudnn
});

const std::string gradient_suffix("_grad");

// Helpers

OperatorDef* NetUtil::AddOp(const std::string& name, const std::vector<std::string>& inputs, const std::vector<std::string>& outputs) {
  auto op = net_.add_op();
  op->set_type(name);
  for (auto input: inputs) {
    op->add_input(input);
  }
  for (auto output: outputs) {
    op->add_output(output);
  }
  return op;
}

Argument* net_add_arg(OperatorDef& op, const std::string& name) {
  auto arg = op.add_arg();
  arg->set_name(name);
  return arg;
}

Argument* net_add_arg(OperatorDef& op, const std::string& name, int value) {
  auto arg = net_add_arg(op, name);
  arg->set_i(value);
  return arg;
}

Argument* net_add_arg(OperatorDef& op, const std::string& name, float value) {
  auto arg = net_add_arg(op, name);
  arg->set_f(value);
  return arg;
}

Argument* net_add_arg(OperatorDef& op, const std::string& name, std::vector<int> values) {
  auto arg = net_add_arg(op, name);
  for (auto value: values) {
    arg->add_ints(value);
  }
  return arg;
}

Argument* net_add_arg(OperatorDef& op, const std::string& name, std::vector<TIndex> values) {
  auto arg = net_add_arg(op, name);
  for (auto value: values) {
    arg->add_ints(value);
  }
  return arg;
}

Argument* net_add_arg(OperatorDef& op, const std::string& name, std::vector<float> values) {
  auto arg = net_add_arg(op, name);
  for (auto value: values) {
    arg->add_floats(value);
  }
  return arg;
}

Argument* net_add_arg(OperatorDef& op, const std::string& name, const std::string& value) {
  auto arg = net_add_arg(op, name);
  arg->set_s(value);
  return arg;
}

Argument* net_add_arg(OperatorDef& op, const std::string& name, const std::vector<std::string>& values) {
  auto arg = net_add_arg(op, name);
  for (auto value: values) {
    arg->add_strings(value);
  }
  return arg;
}

// I/O

OperatorDef* NetUtil::AddCreateDbOp(const std::string& reader, const std::string& db_type, const std::string& db_path) {
  auto op = AddOp("CreateDB", {}, { reader });
  net_add_arg(*op, "db_type", db_type);
  net_add_arg(*op, "db", db_path);
  return op;
}

OperatorDef* NetUtil::AddTensorProtosDbInputOp(const std::string& reader, const std::string& data, const std::string& label, int batch_size) {
  auto op = AddOp("TensorProtosDBInput", { reader }, { data, label });
  net_add_arg(*op, "batch_size", batch_size);
  return op;
}

OperatorDef* NetUtil::AddCoutOp(const std::vector<std::string>& params) {
  return AddOp("Cout", params, {});
}

OperatorDef* NetUtil::AddZeroOneOp(const std::string& pred, const std::string& label) {
  auto op = AddOp("ZeroOne", { pred, label }, {});
  return op;
}

OperatorDef* NetUtil::AddShowWorstOp(const std::string& pred, const std::string& label, const std::string& data) {
  auto op = AddOp("ShowWorst", { pred, label, data }, {});
  return op;
}

OperatorDef* NetUtil::AddEnsureCpuOutputOp(const std::string& input, const std::string& output) {
  auto op = AddOp("EnsureCPUOutput", { input }, { output });
  return op;
}

OperatorDef* NetUtil::AddCopyFromCpuInputOp(const std::string& input, const std::string& output) {
  auto op = AddOp("CopyFromCPUInput", { input }, { output });
  return op;
}

// Initialization

OperatorDef* NetUtil::AddConstantFillOp(const std::vector<int>& shape, const std::string& param) {
  auto op = AddOp("ConstantFill", {}, { param });
  net_add_arg(*op, "shape", shape);
  return op;
}

OperatorDef* NetUtil::AddXavierFillOp(const std::vector<int>& shape, const std::string& param) {
  auto op = AddOp("XavierFill", {}, { param });
  net_add_arg(*op, "shape", shape);
  return op;
}

OperatorDef* NetUtil::AddUniformFillOp(const std::vector<int>& shape, float min, float max, const std::string& param) {
  auto op = AddOp("UniformFill", {}, { param });
  net_add_arg(*op, "shape", shape);
  net_add_arg(*op, "min", min);
  net_add_arg(*op, "max", max);
  return op;
}

OperatorDef* NetUtil::AddConstantFillOp(const std::vector<int>& shape, float value, const std::string& param) {
  auto op = AddOp("ConstantFill", {}, { param });
  net_add_arg(*op, "shape", shape);
  net_add_arg(*op, "value", value);
  return op;
}

OperatorDef* NetUtil::AddConstantFillOp(const std::vector<int>& shape, int64_t value, const std::string& param) {
  auto op = AddOp("ConstantFill", {}, { param });
  net_add_arg(*op, "shape", shape);
  net_add_arg(*op, "value", (int)value);
  net_add_arg(*op, "dtype", TensorProto_DataType_INT64);
  return op;
}

OperatorDef* NetUtil::AddConstantFillWithOp(float value, const std::string& input, const std::string& output) {
  auto op = AddOp("ConstantFill", { input }, { output });
  net_add_arg(*op, "value", value);
  return op;
}

OperatorDef* NetUtil::AddVectorFillOp(const std::vector<int>& values, const std::string& name) {
  auto op = AddOp("GivenTensorFill", {}, { name });
  net_add_arg(*op, "shape", std::vector<int>({ (int)values.size() }));
  net_add_arg(*op, "values", values);
  net_add_arg(*op, "dtype", TensorProto_DataType_INT32);
  return op;
}

OperatorDef* NetUtil::AddGivenTensorFillOp(const TensorCPU& tensor, const std::string& name) {
  auto op = AddOp("XavierFill", {}, { name });
  net_add_arg(*op, "shape", tensor.dims());
  auto arg = net_add_arg(*op, "values");
  const auto& data = tensor.data<float>();
  for (auto i = 0; i < tensor.size(); ++i) {
    arg->add_floats(data[i]);
  }
  return op;
}

// Prediction

OperatorDef* NetUtil::AddConvOp(const std::string& input, const std::string& w, const std::string& b, const std::string& output, int stride, int padding, int kernel) {
  auto op = AddOp("Conv", { input, w, b }, { output });
  net_add_arg(*op, "stride", stride);
  net_add_arg(*op, "pad", padding);
  net_add_arg(*op, "kernel", kernel);
  return op;
}

OperatorDef* NetUtil::AddReluOp(const std::string& input, const std::string& output) {
  return AddOp("Relu", { input }, { output });
}

OperatorDef* NetUtil::AddLrnOp(const std::string& input, const std::string& output, int size, float alpha, float beta, float bias, const std::string& order) {
  auto op = AddOp("LRN", { input }, { output, "_" + output + "_scale" });
  net_add_arg(*op, "size", size);
  net_add_arg(*op, "alpha", alpha);
  net_add_arg(*op, "beta", beta);
  net_add_arg(*op, "bias", bias);
  net_add_arg(*op, "order", order);
  return op;
}

OperatorDef* NetUtil::AddMaxPoolOp(const std::string& input, const std::string& output, int stride, int padding, int kernel, const std::string& order) {
  auto op = AddOp("MaxPool", { input }, { output });
  net_add_arg(*op, "stride", stride);
  net_add_arg(*op, "pad", padding);
  net_add_arg(*op, "kernel", kernel);
  net_add_arg(*op, "order", order);
  net_add_arg(*op, "legacy_pad", 3);
  return op;
}

OperatorDef* NetUtil::AddAveragePoolOp(const std::string& input, const std::string& output, int stride, int padding, int kernel, const std::string& order) {
  auto op = AddOp("AveragePool", { input }, { output });
  net_add_arg(*op, "stride", stride);
  net_add_arg(*op, "pad", padding);
  net_add_arg(*op, "kernel", kernel);
  net_add_arg(*op, "order", order);
  net_add_arg(*op, "legacy_pad", 3);
  return op;
}

OperatorDef* NetUtil::AddFcOp(const std::string& input, const std::string& w, const std::string& b, const std::string& output) {
  return AddOp("FC", { input, w, b }, { output });
}

OperatorDef* NetUtil::AddDropoutOp(const std::string& input, const std::string& output, float ratio) {
  auto op = AddOp("Dropout", { input }, { output, "_" + output + "_mask" });
  net_add_arg(*op, "ratio", ratio);
  net_add_arg(*op, "is_test", 1); // TODO
  return op;
}

OperatorDef* NetUtil::AddSoftmaxOp(const std::string& input, const std::string& output) {
  return AddOp("Softmax", { input }, { output });
}

OperatorDef* NetUtil::AddConcatOp(const std::vector<std::string>& inputs, const std::string& output, const std::string& order) {
  auto op = AddOp("Concat", inputs, { output, "_" + output + "_dims" });
  net_add_arg(*op, "order", order);
  return op;
}

// Training

OperatorDef* NetUtil::AddAccuracyOp(const std::string& pred, const std::string& label, const std::string& accuracy, int top_k) {
  auto op = AddOp("Accuracy", { pred, label }, { accuracy });
  if (top_k) {
    net_add_arg(*op, "top_k", top_k);
  }
  return op;
}

OperatorDef* NetUtil::AddLabelCrossEntropyOp(const std::string& pred, const std::string& label, const std::string& xent) {
  return AddOp("LabelCrossEntropy", { pred, label }, { xent });
}

OperatorDef* NetUtil::AddAveragedLoss(const std::string& input, const std::string& loss) {
  return AddOp("AveragedLoss", { input }, { loss });
}

OperatorDef* NetUtil::AddDiagonalOp(const std::string& input, const std::string& diagonal, const std::vector<int>& offset) {
  auto op = AddOp("Diagonal", { input }, { diagonal });
  net_add_arg(*op, "offset", offset);
  return op;
}

OperatorDef* NetUtil::AddBackMeanOp(const std::string& input, const std::string& mean, int count) {
  auto op = AddOp("BackMean", { input }, { mean });
  net_add_arg(*op, "count", count);
  return op;
}

OperatorDef* NetUtil::AddMeanStdevOp(const std::string& input, const std::string& mean, const std::string& scale) {
  return AddOp("MeanStdev", { input }, { mean, scale });
}

OperatorDef* NetUtil::AddAffineScaleOp(const std::string& input, const std::string& mean, const std::string& scale, const std::string& transformed, bool inverse) {
  auto op = AddOp("AffineScale", { input, mean, scale }, { transformed });
  net_add_arg(*op, "inverse", inverse);
  return op;
}

OperatorDef* NetUtil::AddSliceOp(const std::string& input, const std::string& output, const std::vector<std::pair<int, int>>& ranges) {
  auto op = AddOp("Slice", { input }, { output });
  auto starts = net_add_arg(*op, "starts");
  auto ends = net_add_arg(*op, "ends");
  for (auto r: ranges) {
    starts->add_ints(r.first);
    ends->add_ints(r.second);
  }
  return op;
}

OperatorDef* NetUtil::AddReshapeOp(const std::string& input, const std::string& output, const std::vector<int>& shape) {
  auto op = AddOp("Reshape", { input }, { output, "_" });
  net_add_arg(*op, "shape", shape);
  return op;
}

OperatorDef* NetUtil::AddWeightedSumOp(const std::vector<std::string>& inputs, const std::string& sum) {
  return AddOp("WeightedSum", inputs, { sum });
}

OperatorDef* NetUtil::AddMomentumSgdOp(const std::string& param, const std::string& moment, const std::string& grad, const std::string& lr) {
  return AddOp("MomentumSGDUpdate", { grad, moment, lr, param }, { grad, moment, param });
}

OperatorDef* NetUtil::AddAdagradOp(const std::string& param, const std::string& moment, const std::string& grad, const std::string& lr) {
  return AddOp("Adagrad", { param, moment, grad, lr }, { param, moment });
}

OperatorDef* NetUtil::AddAdamOp(const std::string& param, const std::vector<std::string>& moments, const std::string& grad, const std::string& lr, const std::string& iter) {
  auto op = AddOp("Adam", { param }, { param });
  for (auto& moment: moments) {
    op->add_input(moment);
  }
  op->add_input(grad);
  op->add_input(lr);
  op->add_input(iter);
  for (auto& moment: moments) {
    op->add_output(moment);
  }
  return op;
}

OperatorDef* NetUtil::AddScaleOp(const std::string& input, const std::string& output, float scale) {
  auto op = AddOp("Scale", { input }, { output });
  net_add_arg(*op, "scale", scale);
  return op;
}

OperatorDef* NetUtil::AddClipOp(const std::string& input, const std::string& output, float min, float max) {
  auto op = AddOp("Clip", { input }, { output });
  net_add_arg(*op, "min", min);
  net_add_arg(*op, "max", max);
  return op;
}

OperatorDef* NetUtil::AddCastOp(const std::string& input, const std::string& output, TensorProto::DataType type) {
  auto op = AddOp("Cast", { input }, { output });
  net_add_arg(*op, "to", type);
  return op;
}

OperatorDef* NetUtil::AddIterOp(const std::string& iter) {
  return AddOp("Iter", { iter }, { iter });
}

OperatorDef* NetUtil::AddLearningRateOp(const std::string& iter, const std::string& rate, float base_rate) {
  auto op = AddOp("LearningRate", { iter }, { rate });
  net_add_arg(*op, "policy", "step");
  net_add_arg(*op, "stepsize", 1);
  net_add_arg(*op, "base_lr", -base_rate);
  net_add_arg(*op, "gamma", 0.999f);
  return op;
}

void NetUtil::AddInput(const std::string input) {
  net_.add_external_input(input);
}

void NetUtil::AddOutput(const std::string output) {
  net_.add_external_output(output);
}

void NetUtil::SetName(const std::string name) {
  net_.set_name(name);
}

void NetUtil::SetFillToTrain() {
  for (auto &op: *net_.mutable_op()) {
    if (op.type() == "GivenTensorFill") {
      op.mutable_arg()->RemoveLast();
      if (op.output(0).find("_w") != std::string::npos) {
        op.set_type("XavierFill");
      }
      if (op.output(0).find("_b") != std::string::npos) {
        op.set_type("ConstantFill");
      }
    }
    op.clear_name();
  }
}

void NetUtil::SetRenameInplace() {
  std::set<std::string> renames;
  for (auto &op: *net_.mutable_op()) {
    if (renames.find(op.input(0)) != renames.end()) {
      op.set_input(0, op.input(0) + "_unique");
    }
    if (renames.find(op.output(0)) != renames.end()) {
      renames.erase(op.output(0));
    }
    if (op.input(0) == op.output(0)) {
      if (non_inplace_ops.find(op.type()) != non_inplace_ops.end()) {
        renames.insert(op.output(0));
        op.set_output(0, op.output(0) + "_unique");
      }
    }
  }
}

void NetUtil::SetEngineOps(const std::string engine) {
  for (auto &op: *net_.mutable_op()) {
    op.set_engine(engine);
  }
}

// Gradient

OperatorDef *NetUtil::AddGradientOp(OperatorDef &op) {
  auto grad = net_.add_op();
  if (custom_gradient.find(op.type()) == custom_gradient.end()) {
    vector<GradientWrapper> output(op.output_size());
    for (auto i = 0; i < output.size(); i++) {
      output[i].dense_ = op.output(i) + gradient_suffix;
    }
    GradientOpsMeta meta = GetGradientForOp(op, output);
    grad->CopyFrom(meta.ops_[0]);
  } else {
    grad->set_type(custom_gradient.at(op.type()));
    for (auto arg: op.arg()) {
      auto copy = grad->add_arg();
      copy->CopyFrom(arg);
    }
    for (auto output: op.output()) {
      grad->add_input(output + gradient_suffix);
    }
    for (auto input: op.input()) {
      grad->add_output(input + gradient_suffix);
    }
  }
  grad->set_is_gradient_op(true);
  return grad;
}

void NetUtil::AddGradientOps() {
  for (auto op: CollectGradientOps()) {
    AddGradientOp(op);
  }
}

// Collectors

std::map<std::string, int> NetUtil::CollectParamSizes() {
  std::map<std::string, int> sizes;
  for (const auto &op: net_.op()) {
    if (filler_ops.find(op.type()) != filler_ops.end()) {
      for (const auto &arg: op.arg()) {
        if (arg.name() == "shape") {
          auto size = 1;
          for (auto i: arg.ints()) {
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
  std::set<std::string> external_inputs(net_.external_input().begin(), net_.external_input().end());
  for (const auto &op: net_.op()) {
    if (trainable_ops.find(op.type()) != trainable_ops.end()) {
      for (const auto &input: op.input()) {
        if (external_inputs.find(input) != external_inputs.end()) {
          params.push_back(input);
        }
      }
    }
  }
  return params;
}

std::vector<OperatorDef> NetUtil::CollectGradientOps() {
  std::set<std::string> external_inputs(net_.external_input().begin(), net_.external_input().end());
  std::vector<OperatorDef> gradient_ops;
  for (auto &op: net_.op()) {
    if (trainable_ops.find(op.type()) != trainable_ops.end()) {
      gradient_ops.push_back(op);
      // std::cout << "type: " << op.type() << std::endl;
    } else if (non_trainable_ops.find(op.type()) == non_trainable_ops.end()) {
      std::cout << "unknown backprop operator type: " << op.type() << std::endl;
    }
  }
  std::reverse(gradient_ops.begin(), gradient_ops.end());
  return gradient_ops;
}

std::set<std::string> NetUtil::CollectLayers(const std::string &layer, bool forward) {
  std::map<std::string, std::set<std::string>> lookup;
  for (auto &op: net_.op()) {
    for (auto &input: op.input()) {
      for (auto &output: op.output()) {
        lookup[forward ? input : output].insert(forward ? output : input);
      }
    }
  }
  std::set<std::string> result;
  for (std::set<std::string> step({ layer }); step.size();) {
    std::set<std::string> next;
    for (auto &l: step) {
      if (result.find(l) == result.end()) {
        result.insert(l);
        for (auto &n: lookup[l]) {
          next.insert(n);
        }
      }
    }
    step = next;
  }
  return result;
}

void NetUtil::CheckLayerAvailable(const std::string &layer) {
  std::vector<std::pair<std::string, std::string>> available_layers({ { net_.external_input(0), "Input" } });
  auto layer_found = (net_.external_input(0) == layer);
  for (const auto &op: net_.op()) {
    if (op.input(0) != op.output(0)) {
      available_layers.push_back({ op.output(0), op.type() });
      layer_found |= (op.output(0) == layer);
    }
  }
  if (!layer_found) {
    std::cout << "available layers:" << std::endl;
    for (auto &layer: available_layers) {
      std::cout << "  " << layer.first << " (" << layer.second << ")" << std::endl;
    }
    LOG(FATAL) << "~ no layer with name " << layer << " in model.";
  }
}

}  // namespace caffe2
