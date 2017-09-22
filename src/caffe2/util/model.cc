#include "caffe2/util/model.h"

namespace caffe2 {

const std::string gradient_suffix("_grad");
const std::string moment_suffix("_moment");
const std::string meansq_suffix("_meansq");
const std::string reader_suffix("_reader");
const std::string iter_name("iter");
const std::string lr_name("lr");
const std::string one_name("one");
const std::string loss_name("loss");
const std::string label_name("label");
const std::string xent_name("xent");
const std::string accuracy_name("accuracy");

void ModelUtil::AddDatabaseOps(const std::string &name, const std::string &data,
                               const std::string &db,
                               const std::string &db_type, int batch_size) {
  auto reader = name + reader_suffix;
  init_.AddCreateDbOp(reader, db_type, db);
  predict_.AddInput(reader);
  predict_.AddTensorProtosDbInputOp(reader, data, label_name, batch_size);
  // predict_.AddCoutOp(data);
  // predict_.AddCoutOp(label_name);
}

void ModelUtil::AddTrainOps(const std::string &output, float base_rate,
                            std::string &optimizer) {
  AddXentOps(output);
  predict_.AddConstantFillWithOp(1.0, loss_name, loss_name + gradient_suffix);
  predict_.AddGradientOps();
  AddIterLrOps(base_rate);
  AddOptimizerOps(optimizer);
}

void ModelUtil::AddTestOps(const std::string &output) { AddXentOps(output); }

void ModelUtil::AddXentOps(const std::string &output) {
  predict_.AddLabelCrossEntropyOp(output, label_name, xent_name);
  predict_.AddAveragedLossOp(xent_name, loss_name);
  predict_.AddAccuracyOp(output, label_name, accuracy_name);
}

void ModelUtil::AddIterLrOps(float base_rate) {
  init_.AddConstantFillOp({1}, (int64_t)0, iter_name)
      ->mutable_device_option()
      ->set_device_type(CPU);
  predict_.AddInput(iter_name);
  predict_.AddIterOp(iter_name);
  predict_.AddLearningRateOp(iter_name, lr_name, base_rate);
}

void ModelUtil::AddSgdOps() {
  init_.AddConstantFillOp({1}, 1.f, one_name);
  predict_.AddInput(one_name);
  for (auto &param : predict_.CollectParams()) {
    predict_.AddWeightedSumOp(
        {param, one_name, param + gradient_suffix, lr_name}, param);
  }
}

void ModelUtil::AddMomentumOps() {
  auto sizes = init_.CollectParamSizes();
  for (auto &param : predict_.CollectParams()) {
    auto size = sizes[param];
    init_.AddConstantFillOp({size}, 0.f, param + moment_suffix);
    predict_.AddInput(param + moment_suffix);
    predict_.AddMomentumSgdOp(param, param + moment_suffix,
                              param + gradient_suffix, lr_name);
  }
}

void ModelUtil::AddAdagradOps() {
  auto sizes = init_.CollectParamSizes();
  for (auto &param : predict_.CollectParams()) {
    auto size = sizes[param];
    init_.AddConstantFillOp({size}, 0.f, param + moment_suffix);
    predict_.AddInput(param + moment_suffix);
    predict_.AddAdagradOp(param, param + moment_suffix, param + gradient_suffix,
                          lr_name);
  }
}

void ModelUtil::AddAdamOps() {
  auto sizes = init_.CollectParamSizes();
  for (auto &param : predict_.CollectParams()) {
    auto size = sizes[param];
    std::vector<std::string> moments(2);
    auto i = 0;
    for (auto &moment : moments) {
      moment = param + moment_suffix + "_" + std::to_string(++i);
      init_.AddConstantFillOp({size}, 0.f, moment);
      predict_.AddInput(moment);
    }
    predict_.AddAdamOp(param, moments, param + gradient_suffix, lr_name,
                       iter_name);
  }
}

void ModelUtil::AddRmsPropOps() {
  auto sizes = init_.CollectParamSizes();
  for (auto &param : predict_.CollectParams()) {
    auto size = sizes[param];
    auto moment_name = param + moment_suffix;
    auto meansq_name = param + meansq_suffix;
    init_.AddConstantFillOp({size}, 0.f, moment_name);
    init_.AddConstantFillOp({size}, 0.f, meansq_name);
    predict_.AddInput(moment_name);
    predict_.AddInput(meansq_name);
    predict_.AddRmsPropOp(param + gradient_suffix, meansq_name, moment_name,
                          lr_name);
    predict_.AddSumOp({param, param + gradient_suffix}, param);
  }
}

void ModelUtil::AddOptimizerOps(std::string &optimizer) {
  if (optimizer == "sgd") {
    AddSgdOps();
  } else if (optimizer == "momentum") {
    AddMomentumOps();
  } else if (optimizer == "adagrad") {
    AddAdagradOps();
  } else if (optimizer == "adam") {
    AddAdamOps();
  } else if (optimizer == "rmsprop") {
    AddRmsPropOps();
  } else {
    LOG(FATAL) << "~ optimizer type not supported: " << optimizer;
  }
}

}  // namespace caffe2
