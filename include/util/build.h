#ifndef BUILD_H
#define BUILD_H

#include "caffe2/core/net.h"

namespace caffe2 {

static const std::set<std::string> trainable_ops({
  "Add",
  "AveragedLoss",
  "Concat",
  "Conv",
  "FC",
  "LabelCrossEntropy",
  "LRN",
  "MaxPool",
  "Mul",
  "Relu",
  "Softmax",
  "SpatialBN",
  "Sum",
});

static const std::set<std::string> non_trainable_ops({
  "Accuracy",
  "AveragePool",
  "Cout",
  "ConstantFill",
  "Dropout",
  "TensorProtosDBInput",
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

OperatorDef *add_cout_op(NetDef &model, const std::string &param) {
  auto op = model.add_op();
  op->set_type("Cout");
  auto arg = op->add_arg();
  op->add_input(param);
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

OperatorDef *add_averaged_loss(NetDef &model, const std::string &xent, const std::string &loss) {
  auto op = model.add_op();
  op->set_type("AveragedLoss");
  op->add_input(xent);
  op->add_output(loss);
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

OperatorDef *add_adam_op(NetDef &model, const std::string &param, const std::string &m1, const std::string &m2, const std::string &grad, const std::string &lr, const std::string &iter) {
  auto op = model.add_op();
  op->set_type("Adam");
  op->add_input(param);
  op->add_input(m1);
  op->add_input(m2);
  op->add_input(grad);
  op->add_input(lr);
  op->add_input(iter);
  op->add_output(param);
  op->add_output(m1);
  op->add_output(m2);
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

OperatorDef *add_constant_fill_op(NetDef &model, const std::vector<int> &shape, const std::string &param) {
  auto op = model.add_op();
  op->set_type("ConstantFill");
  auto arg = op->add_arg();
  arg->set_name("shape");
  for (auto dim: shape) {
    arg->add_ints(dim);
  }
  op->add_output(param);
  return op;
}

OperatorDef *add_constant_fill_float_op(NetDef &model, const std::vector<int> &shape, float value, const std::string &param) {
  auto op = add_constant_fill_op(model, shape, param);
  auto arg = op->add_arg();
  arg->set_name("value");
  arg->set_f(value);
  return op;
}

OperatorDef *add_constant_fill_int64_op(NetDef &model, const std::vector<int> &shape, int64_t value, const std::string &param) {
  auto op = add_constant_fill_op(model, shape, param);
  auto arg1 = op->add_arg();
  arg1->set_name("value");
  arg1->set_i(value);
  auto arg2 = op->add_arg();
  arg2->set_name("dtype");
  arg2->set_i(TensorProto_DataType_INT64);
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
  op->set_is_gradient_op(true);
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

OperatorDef *add_gradient_op(NetDef &model, const OperatorDef *op) {
  vector<GradientWrapper> output(op->output_size());
  for (auto i = 0; i < output.size(); i++) {
    output[i].dense_ = op->output(i) + gradient_suffix;
  }
  GradientOpsMeta meta = GetGradientForOp(*op, output);
  auto grad = model.add_op();
  grad->CopyFrom(meta.ops_[0]);
  grad->set_is_gradient_op(true);
  return grad;
}

void add_gradient_ops(NetDef &model) {
  std::set<std::string> external_inputs(model.external_input().begin(), model.external_input().end());
  auto x = model.op();
  for (auto i = x.rbegin(); i != x.rend(); ++i) {
    if (trainable_ops.find(i->type()) != trainable_ops.end()) {
      add_gradient_op(model, &*i);
      // std::cout << "type: " << op.type() << std::endl;
    } else if (non_trainable_ops.find(i->type()) == non_trainable_ops.end()) {
      std::cout << "unknown backprop operator type: " << i->type() << std::endl;
    }
  }
}

void add_database_ops(NetDef &init_model, NetDef &predict_model, const std::string &name, const std::string &data, const std::string &db, const std::string &db_type, int batch_size) {
  auto reader = name + "_dbreader";
  add_create_db_op(init_model, reader, db_type, db);
  predict_model.add_external_input(reader);
  add_tensor_protos_db_input_op(predict_model, reader, data, "label", batch_size);
  // add_cout_op(predict_model, data);
  // add_cout_op(predict_model, "label");
}

void add_test_ops(NetDef &model) {
  add_accuracy_op(model, "prob", "label", "accuracy");
}

void add_xent_ops(NetDef &model) {
  add_label_cross_entropy_op(model, "prob", "label", "xent");
  add_averaged_loss(model, "xent", "loss");
  add_accuracy_op(model, "prob", "label", "accuracy");
  add_constant_fill_with_op(model, 1.0, "loss", "loss" + gradient_suffix);
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
  set_device_cpu_op(*add_constant_fill_int64_op(init_model, { 1 }, 0, "iter"));
  predict_model.add_external_input("iter");
  add_iter_op(predict_model, "iter");
  add_learning_rate_op(predict_model, "iter", "LR", base_rate);
}

void add_sgd_ops(NetDef &init_model, NetDef &predict_model) {
  add_constant_fill_float_op(init_model, { 1 }, 1.0, "ONE");
  predict_model.add_external_input("ONE");
  for (auto &param: collect_params(predict_model)) {
    add_weighted_sum_op(predict_model, { param, "ONE", param + gradient_suffix, "LR" }, param);
  }
}

void add_adam_ops(NetDef &init_model, NetDef &predict_model) {
  auto sizes = collect_param_sizes(init_model);
  for (auto &param: collect_params(predict_model)) {
    auto size = sizes[param];
    add_constant_fill_float_op(init_model, { size }, 0.0, param + "_m1");
    add_constant_fill_float_op(init_model, { size }, 0.0, param + "_m2");
    predict_model.add_external_input(param + "_m1");
    predict_model.add_external_input(param + "_m2");
    add_adam_op(predict_model, param, param + "_m1", param + "_m2", param + gradient_suffix, "LR", "iter");
  }
}

void add_optimizer_ops(NetDef &init_model, NetDef &predict_model, std::string &optimizer) {
  if (optimizer == "sgd") {
    add_sgd_ops(init_model, predict_model);
  } else if (optimizer == "adam") {
    add_adam_ops(init_model, predict_model);
  } else {
    LOG(FATAL) << "~ optimizer type not supported: " << optimizer;
  }
}

void add_train_ops(NetDef &init_model, NetDef &predict_model, float base_rate, std::string &optimizer) {
  add_xent_ops(predict_model);
  add_gradient_ops(predict_model);
  add_iter_lr_ops(init_model, predict_model, base_rate);
  add_optimizer_ops(init_model, predict_model, optimizer);
}

}  // namespace caffe2

#endif  // BUILD_H
