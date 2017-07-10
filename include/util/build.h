#ifndef BUILD_H
#define BUILD_H

#include "caffe2/core/net.h"
#include "caffe2/core/operator_gradient.h"

namespace caffe2 {

static const std::set<std::string> trainable_ops({
  "Add",
  "AffineScale",
  "AveragedLoss",
  "AveragePool",
  "BackMean",
  "Concat",
  "Conv",
  "Diagonal",
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

static const std::set<std::string> non_trainable_ops({
  "Accuracy",
  "Cout",
  "ConstantFill",
  "Dropout",
  "TensorProtosDBInput",
});

static const std::map<std::string, std::string> custom_gradient({
  { "EnsureCPUOutput", "CopyFromCPUInput" },
  { "CopyFromCPUInput", "EnsureCPUOutput" },
});

static const std::set<std::string> filler_ops({
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

static const std::string gradient_suffix("_grad");
static const std::string moment_suffix("_moment");
static const std::string reader_suffix("_reader");
static const std::string iter_name("iter");
static const std::string lr_name("lr");
static const std::string one_name("one");
static const std::string loss_name("loss");
static const std::string label_name("label");
static const std::string xent_name("xent");
static const std::string accuracy_name("accuracy");

// Operators

OperatorDef *add_create_db_op(NetDef &model, const std::string &reader, const std::string &db_type, const std::string &db_path) {
  auto op = model.add_op();
  op->set_type("CreateDB");
  auto arg1 = op->add_arg();
  arg1->set_name("db_type");
  arg1->set_s(db_type);
  auto arg2 = op->add_arg();
  arg2->set_name("db");
  arg2->set_s(db_path);
  op->add_output(reader);
  return op;
}

OperatorDef *add_tensor_protos_db_input_op(NetDef &model, const std::string &reader, const std::string &data, const std::string &label, int batch_size) {
  auto op = model.add_op();
  op->set_type("TensorProtosDBInput");
  auto arg = op->add_arg();
  arg->set_name("batch_size");
  arg->set_i(batch_size);
  op->add_input(reader);
  op->add_output(data);
  op->add_output(label);
  return op;
}

OperatorDef *add_cout_op(NetDef &model, const std::vector<std::string> &params) {
  auto op = model.add_op();
  op->set_type("Cout");
  for (auto param: params) {
    op->add_input(param);
  }
  return op;
}

OperatorDef *add_accuracy_op(NetDef &model, const std::string &pred, const std::string &label, const std::string &accuracy) {
  auto op = model.add_op();
  op->set_type("Accuracy");
  op->add_input(pred);
  op->add_input(label);
  op->add_output(accuracy);
  return op;
}

OperatorDef *add_label_cross_entropy_op(NetDef &model, const std::string &pred, const std::string &label, const std::string &xent) {
  auto op = model.add_op();
  op->set_type("LabelCrossEntropy");
  op->add_input(pred);
  op->add_input(label);
  op->add_output(xent);
  return op;
}

OperatorDef *add_averaged_loss(NetDef &model, const std::string &input, const std::string &loss) {
  auto op = model.add_op();
  op->set_type("AveragedLoss");
  op->add_input(input);
  op->add_output(loss);
  return op;
}

OperatorDef *add_ensure_cpu_output_op(NetDef &model, const std::string &input, const std::string &output) {
  auto op = model.add_op();
  op->set_type("EnsureCPUOutput");
  op->add_input(input);
  op->add_output(output);
  return op;
}

OperatorDef *add_copy_from_cpu_input_op(NetDef &model, const std::string &input, const std::string &output) {
  auto op = model.add_op();
  op->set_type("CopyFromCPUInput");
  op->add_input(input);
  op->add_output(output);
  return op;
}

OperatorDef *add_diagonal_op(NetDef &model, const std::string &input, const std::string &diagonal, const std::vector<int> &offset) {
  auto op = model.add_op();
  op->set_type("Diagonal");
  auto arg = op->add_arg();
  arg->set_name("offset");
  for (auto o: offset) {
    arg->add_ints(o);
  }
  op->add_input(input);
  op->add_output(diagonal);
  op->mutable_device_option()->set_device_type(CPU);
  return op;
}

OperatorDef *add_back_mean_op(NetDef &model, const std::string &input, const std::string &mean, int count = 1) {
  auto op = model.add_op();
  op->set_type("BackMean");
  auto arg = op->add_arg();
  arg->set_name("count");
  arg->set_i(count);
  op->add_input(input);
  op->add_output(mean);
  op->mutable_device_option()->set_device_type(CPU);
  return op;
}

OperatorDef *add_mean_stdev_op(NetDef &model, const std::string &input, const std::string &mean, const std::string &scale) {
  auto op = model.add_op();
  op->set_type("MeanStdev");
  op->add_input(input);
  op->add_output(mean);
  op->add_output(scale);
  op->mutable_device_option()->set_device_type(CPU);
  return op;
}

OperatorDef *add_affine_scale_op(NetDef &model, const std::string &input, const std::string &mean, const std::string &scale, const std::string &transformed, bool inverse = false) {
  auto op = model.add_op();
  op->set_type("AffineScale");
  auto arg = op->add_arg();
  arg->set_name("inverse");
  arg->set_i(inverse);
  op->add_input(input);
  op->add_input(mean);
  op->add_input(scale);
  op->add_output(transformed);
  op->mutable_device_option()->set_device_type(CPU);
  return op;
}

OperatorDef *add_slice_op(NetDef &model, const std::string &input, const std::string &output, const std::vector<std::pair<int, int>> &ranges) {
  auto op = model.add_op();
  op->set_type("Slice");
  auto arg1 = op->add_arg();
  arg1->set_name("starts");
  auto arg2 = op->add_arg();
  arg2->set_name("ends");
  for (auto r: ranges) {
    arg1->add_ints(r.first);
    arg2->add_ints(r.second);
  }
  op->add_input(input);
  op->add_output(output);
  return op;
}

OperatorDef *add_reshape_op(NetDef &model, const std::string &input, const std::string &output, const std::vector<int> &shape) {
  auto op = model.add_op();
  op->set_type("Reshape");
  auto arg = op->add_arg();
  arg->set_name("shape");
  for (auto s: shape) {
    arg->add_ints(s);
  }
  op->add_input(input);
  op->add_output(output);
  op->add_output("_");
  return op;
}

OperatorDef *add_weighted_sum_op(NetDef &model, const std::vector<std::string> &inputs, const std::string &sum) {
  auto op = model.add_op();
  op->set_type("WeightedSum");
  for (const auto &input: inputs) {
    op->add_input(input);
  }
  op->add_output(sum);
  return op;
}

OperatorDef *add_momentum_sgd_op(NetDef &model, const std::string &param, const std::string &moment, const std::string &grad, const std::string &lr) {
  auto op = model.add_op();
  op->set_type("MomentumSGDUpdate");
  op->add_input(grad);
  op->add_input(moment);
  op->add_input(lr);
  op->add_input(param);
  op->add_output(grad);
  op->add_output(moment);
  op->add_output(param);
  return op;
}

OperatorDef *add_adagrad_op(NetDef &model, const std::string &param, const std::string &moment, const std::string &grad, const std::string &lr) {
  auto op = model.add_op();
  op->set_type("Adagrad");
  op->add_input(param);
  op->add_input(moment);
  op->add_input(grad);
  op->add_input(lr);
  op->add_output(param);
  op->add_output(moment);
  return op;
}

OperatorDef *add_adam_op(NetDef &model, const std::string &param, const std::vector<std::string> &moments, const std::string &grad, const std::string &lr, const std::string &iter) {
  auto op = model.add_op();
  op->set_type("Adam");
  op->add_input(param);
  for (auto &moment: moments) {
    op->add_input(moment);
  }
  op->add_input(grad);
  op->add_input(lr);
  op->add_input(iter);
  op->add_output(param);
  for (auto &moment: moments) {
    op->add_output(moment);
  }
  return op;
}

OperatorDef *add_scale_op(NetDef &model, const std::string &input, const std::string &output, float scale) {
  auto op = model.add_op();
  op->set_type("Scale");
  auto arg = op->add_arg();
  arg->set_name("scale");
  arg->set_f(scale);
  op->add_input(input);
  op->add_output(output);
  return op;
}

OperatorDef *add_clip_op(NetDef &model, const std::string &input, const std::string &output, float min, float max) {
  auto op = model.add_op();
  op->set_type("Clip");
  auto arg1 = op->add_arg();
  arg1->set_name("min");
  arg1->set_f(min);
  auto arg2 = op->add_arg();
  arg2->set_name("max");
  arg2->set_f(max);
  op->add_input(input);
  op->add_output(output);
  return op;
}

OperatorDef *add_cast_op(NetDef &model, const std::string &input, const std::string &output, TensorProto::DataType type) {
  auto op = model.add_op();
  op->set_type("Cast");
  auto arg = op->add_arg();
  arg->set_name("to");
  arg->set_i(type);
  op->add_input(input);
  op->add_output(output);
  return op;
}

OperatorDef *add_fill_op(NetDef &model, const std::string type, const std::vector<int> &shape, const std::string &param) {
  auto op = model.add_op();
  op->set_type(type);
  auto arg = op->add_arg();
  arg->set_name("shape");
  for (auto dim: shape) {
    arg->add_ints(dim);
  }
  op->add_output(param);
  return op;
}

OperatorDef *add_uniform_fill_float_op(NetDef &model, const std::vector<int> &shape, float min, float max, const std::string &param) {
  auto op = add_fill_op(model, "UniformFill", shape, param);
  auto arg1 = op->add_arg();
  arg1->set_name("min");
  arg1->set_f(min);
  auto arg2 = op->add_arg();
  arg2->set_name("max");
  arg2->set_f(max);
  return op;
}

OperatorDef *add_constant_fill_float_op(NetDef &model, const std::vector<int> &shape, float value, const std::string &param) {
  auto op = add_fill_op(model, "ConstantFill", shape, param);
  auto arg = op->add_arg();
  arg->set_name("value");
  arg->set_f(value);
  return op;
}

OperatorDef *add_constant_fill_int64_op(NetDef &model, const std::vector<int> &shape, int64_t value, const std::string &param) {
  auto op = add_fill_op(model, "ConstantFill", shape, param);
  auto arg1 = op->add_arg();
  arg1->set_name("value");
  arg1->set_i(value);
  auto arg2 = op->add_arg();
  arg2->set_name("dtype");
  arg2->set_i(TensorProto_DataType_INT64);
  return op;
}

OperatorDef *add_constant_fill_int32_op(NetDef &model, const std::vector<int> &shape, int32_t value, const std::string &param) {
  auto op = add_fill_op(model, "ConstantFill", shape, param);
  auto arg1 = op->add_arg();
  arg1->set_name("value");
  arg1->set_i(value);
  auto arg2 = op->add_arg();
  arg2->set_name("dtype");
  arg2->set_i(TensorProto_DataType_INT32);
  return op;
}

OperatorDef *add_constant_fill_with_op(NetDef &model, float value, const std::string &input, const std::string &output) {
  auto op = model.add_op();
  op->set_type("ConstantFill");
  auto arg = op->add_arg();
  arg->set_name("value");
  arg->set_f(value);
  op->add_input(input);
  op->add_output(output);
  return op;
}

OperatorDef *add_vector_fill_op(NetDef &model, const std::vector<int> &values, const std::string &name) {
  auto op = model.add_op();
  op->set_type("GivenTensorFill");
  auto arg1 = op->add_arg();
  arg1->set_name("shape");
  arg1->add_ints(values.size());
  auto arg2 = op->add_arg();
  arg2->set_name("values");
  for (auto v: values) {
    arg2->add_ints(v);
  }
  auto arg3 = op->add_arg();
  arg3->set_name("dtype");
  arg3->set_i(TensorProto_DataType_INT32);
  op->add_output(name);
  return op;
}

OperatorDef *add_given_tensor_fill_op(NetDef &model, const TensorCPU &tensor, const std::string &name) {
  auto op = model.add_op();
  op->set_type("GivenTensorFill");
  auto arg1 = op->add_arg();
  arg1->set_name("shape");
  for (auto dim: tensor.dims()) {
    arg1->add_ints(dim);
  }
  auto arg2 = op->add_arg();
  arg2->set_name("values");
  const auto& data = tensor.data<float>();
  for (auto i = 0; i < tensor.size(); ++i) {
    arg2->add_floats(data[i]);
  }
  op->add_output(name);
  return op;
}

OperatorDef *add_iter_op(NetDef &model, const std::string &iter) {
  auto op = model.add_op();
  op->set_type("Iter");
  op->add_input(iter);
  op->add_output(iter);
  return op;
}

OperatorDef *add_learning_rate_op(NetDef &model, const std::string &iter, const std::string &rate, float base_rate) {
  auto op = model.add_op();
  op->set_type("LearningRate");
  auto arg1 = op->add_arg();
  arg1->set_name("policy");
  arg1->set_s("step");
  auto arg2 = op->add_arg();
  arg2->set_name("stepsize");
  arg2->set_i(1);
  auto arg3 = op->add_arg();
  arg3->set_name("base_lr");
  arg3->set_f(-base_rate);
  auto arg4 = op->add_arg();
  arg4->set_name("gamma");
  arg4->set_f(0.999);
  op->add_input(iter);
  op->add_output(rate);
  return op;
}

// Helpers

void set_device_cpu_op(OperatorDef &op) {
  op.mutable_device_option()->set_device_type(CPU);
}

void set_engine_cudnn_op(OperatorDef &op) {
  op.set_engine("CUDNN");
}

void set_engine_cudnn_net(NetDef &net) {
  for (auto &op: *net.mutable_op()) {
    op.set_engine("CUDNN");
  }
}

OperatorDef *add_gradient_op(NetDef &model, OperatorDef &op) {
  auto grad = model.add_op();
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
      grad->add_input(output);
    }
    for (auto input: op.input()) {
      grad->add_output(input + "_grad");
    }
  }
  grad->set_is_gradient_op(true);
  return grad;
}

std::vector<OperatorDef> collect_gradient_ops(NetDef &model) {
  std::set<std::string> external_inputs(model.external_input().begin(), model.external_input().end());
  std::vector<OperatorDef> gradient_ops;
  for (auto &op: model.op()) {
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

void add_gradient_ops(NetDef &model) {
  for (auto op: collect_gradient_ops(model)) {
    add_gradient_op(model, op);
  }
}

void add_database_ops(NetDef &init_model, NetDef &predict_model, const std::string &name, const std::string &data, const std::string &db, const std::string &db_type, int batch_size) {
  auto reader = name + reader_suffix;
  add_create_db_op(init_model, reader, db_type, db);
  predict_model.add_external_input(reader);
  add_tensor_protos_db_input_op(predict_model, reader, data, label_name, batch_size);
  // add_cout_op(predict_model, data);
  // add_cout_op(predict_model, label_name);
}

void add_test_ops(NetDef &model, const std::string &output) {
  add_accuracy_op(model, output, label_name, accuracy_name);
}

void add_xent_ops(NetDef &model, const std::string &output) {
  add_label_cross_entropy_op(model, output, label_name, xent_name);
  add_averaged_loss(model, xent_name, loss_name);
  add_accuracy_op(model, output, label_name, accuracy_name);
  add_constant_fill_with_op(model, 1.0, loss_name, loss_name + gradient_suffix);
}

std::map<std::string, int> collect_param_sizes(NetDef &model) {
  std::map<std::string, int> sizes;
  for (const auto &op: model.op()) {
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

std::vector<std::string> collect_params(NetDef &model) {
  std::vector<std::string> params;
  std::set<std::string> external_inputs(model.external_input().begin(), model.external_input().end());
  for (const auto &op: model.op()) {
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

void add_iter_lr_ops(NetDef &init_model, NetDef &predict_model, float base_rate) {
  set_device_cpu_op(*add_constant_fill_int64_op(init_model, { 1 }, 0, iter_name));
  predict_model.add_external_input(iter_name);
  add_iter_op(predict_model, iter_name);
  add_learning_rate_op(predict_model, iter_name, lr_name, base_rate);
}

void add_sgd_ops(NetDef &init_model, NetDef &predict_model) {
  add_constant_fill_float_op(init_model, { 1 }, 1.0, one_name);
  predict_model.add_external_input(one_name);
  for (auto &param: collect_params(predict_model)) {
    add_weighted_sum_op(predict_model, { param, one_name, param + gradient_suffix, lr_name }, param);
  }
}

void add_momentum_ops(NetDef &init_model, NetDef &predict_model) {
  auto sizes = collect_param_sizes(init_model);
  for (auto &param: collect_params(predict_model)) {
    auto size = sizes[param];
    add_constant_fill_float_op(init_model, { size }, 0.0, param + moment_suffix);
    predict_model.add_external_input(param + moment_suffix);
    add_momentum_sgd_op(predict_model, param, param + moment_suffix, param + gradient_suffix, lr_name);
  }
}

void add_adagrad_ops(NetDef &init_model, NetDef &predict_model) {
  auto sizes = collect_param_sizes(init_model);
  for (auto &param: collect_params(predict_model)) {
    auto size = sizes[param];
    add_constant_fill_float_op(init_model, { size }, 0.0, param + moment_suffix);
    predict_model.add_external_input(param + moment_suffix);
    add_adagrad_op(predict_model, param, param + moment_suffix, param + gradient_suffix, lr_name);
  }
}

void add_adam_ops(NetDef &init_model, NetDef &predict_model) {
  auto sizes = collect_param_sizes(init_model);
  for (auto &param: collect_params(predict_model)) {
    auto size = sizes[param];
    std::vector<std::string> moments(2);
    auto i = 0;
    for (auto &moment: moments) {
      moment = param + moment_suffix + "_" + std::to_string(++i);
      add_constant_fill_float_op(init_model, { size }, 0.0, moment);
      predict_model.add_external_input(moment);
    }
    add_adam_op(predict_model, param, moments, param + gradient_suffix, lr_name, iter_name);
  }
}

void add_optimizer_ops(NetDef &init_model, NetDef &predict_model, std::string &optimizer) {
  if (optimizer == "sgd") {
    add_sgd_ops(init_model, predict_model);
  } else if (optimizer == "momentum") {
    add_momentum_ops(init_model, predict_model);
  } else if (optimizer == "adagrad") {
    add_adagrad_ops(init_model, predict_model);
  } else if (optimizer == "adam") {
    add_adam_ops(init_model, predict_model);
  } else {
    LOG(FATAL) << "~ optimizer type not supported: " << optimizer;
  }
}

void add_train_ops(NetDef &init_model, NetDef &predict_model, const std::string &output, float base_rate, std::string &optimizer) {
  add_xent_ops(predict_model, output);
  add_gradient_ops(predict_model);
  add_iter_lr_ops(init_model, predict_model, base_rate);
  add_optimizer_ops(init_model, predict_model, optimizer);
}

}  // namespace caffe2

#endif  // BUILD_H
