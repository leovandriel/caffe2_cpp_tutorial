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

void ModelUtil::SetName(const std::string &name) {
  init.SetName(name + "_init");
  predict.SetName(name + "_predict");
}

void ModelUtil::AddDatabaseOps(const std::string &name, const std::string &data,
                               const std::string &db,
                               const std::string &db_type, int batch_size) {
  auto reader = name + reader_suffix;
  init.AddCreateDbOp(reader, db_type, db);
  predict.AddInput(reader);
  predict.AddTensorProtosDbInputOp(reader, data, label_name, batch_size);
  // predict.AddCoutOp(data);
  // predict.AddCoutOp(label_name);
}

void ModelUtil::AddGradientOps() {
  predict.AddConstantFillWithOp(1.0, loss_name, loss_name + gradient_suffix);
  predict.AddAllGradientOp();
}

void ModelUtil::AddTrainOps(const std::string &output, float base_rate,
                            std::string &optimizer) {
  AddXentOps(output);
  AddGradientOps();
  AddIterOps();
  predict.AddLearningRateOp(iter_name, lr_name, base_rate);
  AddOptimizerOps(optimizer);
}

void ModelUtil::AddTestOps(const std::string &output) {
  AddXentOps(output);
  predict.AddInput(iter_name);
}

void ModelUtil::AddXentOps(const std::string &output) {
  predict.AddLabelCrossEntropyOp(output, label_name, xent_name);
  predict.AddAveragedLossOp(xent_name, loss_name);
  predict.AddAccuracyOp(output, label_name, accuracy_name);
}

void ModelUtil::AddIterOps() {
  init.AddConstantFillOp({1}, (int64_t)0, iter_name)
      ->mutable_device_option()
      ->set_device_type(CPU);
  predict.AddInput(iter_name);
  predict.AddIterOp(iter_name);
}

void ModelUtil::AddSgdOps() {
  init.AddConstantFillOp({1}, 1.f, one_name);
  predict.AddInput(one_name);
  for (auto &param : predict.CollectParams()) {
    predict.AddWeightedSumOp(
        {param, one_name, param + gradient_suffix, lr_name}, param);
  }
}

void ModelUtil::AddMomentumOps() {
  auto sizes = init.CollectParamSizes();
  for (auto &param : predict.CollectParams()) {
    auto size = sizes[param];
    init.AddConstantFillOp({size}, 0.f, param + moment_suffix);
    predict.AddInput(param + moment_suffix);
    predict.AddMomentumSgdOp(param, param + moment_suffix,
                             param + gradient_suffix, lr_name);
  }
}

void ModelUtil::AddAdagradOps() {
  auto sizes = init.CollectParamSizes();
  for (auto &param : predict.CollectParams()) {
    auto size = sizes[param];
    init.AddConstantFillOp({size}, 0.f, param + moment_suffix);
    predict.AddInput(param + moment_suffix);
    predict.AddAdagradOp(param, param + moment_suffix, param + gradient_suffix,
                         lr_name);
  }
}

void ModelUtil::AddAdamOps() {
  auto sizes = init.CollectParamSizes();
  for (auto &param : predict.CollectParams()) {
    auto size = sizes[param];
    std::vector<std::string> moments(2);
    auto i = 0;
    for (auto &moment : moments) {
      moment = param + moment_suffix + "_" + std::to_string(++i);
      init.AddConstantFillOp({size}, 0.f, moment);
      predict.AddInput(moment);
    }
    predict.AddAdamOp(param, moments, param + gradient_suffix, lr_name,
                      iter_name);
  }
}

void ModelUtil::AddRmsPropOps() {
  auto sizes = init.CollectParamSizes();
  for (auto &param : predict.CollectParams()) {
    auto size = sizes[param];
    auto moment_name = param + moment_suffix;
    auto meansq_name = param + meansq_suffix;
    init.AddConstantFillOp({size}, 0.f, moment_name);
    init.AddConstantFillOp({size}, 0.f, meansq_name);
    predict.AddInput(moment_name);
    predict.AddInput(meansq_name);
    predict.AddRmsPropOp(param + gradient_suffix, meansq_name, moment_name,
                         lr_name);
    predict.AddSumOp({param, param + gradient_suffix}, param);
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

void ModelUtil::AddFcOps(const std::string &input, const std::string &output,
                         int in_size, int out_size, bool test) {
  if (!test) {
    init.AddXavierFillOp({out_size, in_size}, output + "_w");
    init.AddConstantFillOp({out_size}, output + "_b");
  }
  predict.AddInput(output + "_w");
  predict.AddInput(output + "_b");
  predict.AddFcOp(input, output + "_w", output + "_b", output);
}

void ModelUtil::AddConvOps(const std::string &input, const std::string &output,
                           int in_size, int out_size, int stride, int padding,
                           int kernel, bool test) {
  if (!test) {
    init.AddXavierFillOp({out_size, in_size, kernel, kernel}, output + "_w");
    init.AddConstantFillOp({out_size}, output + "_b");
  }
  predict.AddInput(output + "_w");
  predict.AddInput(output + "_b");
  predict.AddConvOp(input, output + "_w", output + "_b", output, stride,
                    padding, kernel);
}

}  // namespace caffe2
