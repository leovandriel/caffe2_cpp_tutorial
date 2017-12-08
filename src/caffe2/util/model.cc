#include "caffe2/util/model.h"
#include "caffe2/util/blob.h"

namespace caffe2 {

const std::string gradient_suffix("_grad");
const std::string moment_suffix("_moment");
const std::string meansq_suffix("_meansq");
const std::string reader_suffix("_reader");
const std::string init_net_suffix("_init_net.pb");
const std::string predict_net_suffix("_predict_net.pb");
const std::string init_name_suffix("_init");
const std::string predict_name_suffix("_predict");

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

void ModelUtil::AddSpatialBNOp(const std::string &input,
                               const std::string &output, int size,
                               float epsilon, float momentum, bool test) {
  if (!test) {
    init.AddConstantFillOp({size}, 1.0f, output + "_scale");
    init.AddConstantFillOp({size}, 0.0f, output + "_bias");
    init.AddConstantFillOp({size}, 0.0f, output + "_mean");
    init.AddConstantFillOp({size}, 1.0f, output + "_var");
  }
  predict.AddInput(output + "_scale");
  predict.AddInput(output + "_bias");
  if (!test)
    predict.AddSpatialBNOp({input, output + "_scale", output + "_bias",
                            output + "_mean", output + "_var"},
                           {output, output + "_mean", output + "_var",
                            output + "_saved_mean", output + "_saved_var"},
                           epsilon, momentum, test);
  else
    predict.AddSpatialBNOp({input, output + "_scale", output + "_bias"},
                           {output}, epsilon, momentum, test);
}

void ModelUtil::Split(const std::string &layer, ModelUtil &firstModel,
                      ModelUtil &secondModel, bool force_cpu, bool inclusive) {
  std::set<std::string> static_inputs = predict.CollectLayers(layer);

  // copy operators
  for (const auto &op : init.net.op()) {
    auto is_first = (static_inputs.find(op.output(0)) != static_inputs.end());
    auto new_op =
        (is_first ? firstModel.init.net : secondModel.init.net).add_op();
    new_op->CopyFrom(op);
  }
  for (const auto &op : predict.net.op()) {
    auto is_first = (static_inputs.find(op.output(0)) != static_inputs.end() &&
                     (inclusive || op.input(0) != op.output(0)));
    auto new_op =
        (is_first ? firstModel.predict.net : secondModel.predict.net).add_op();
    new_op->CopyFrom(op);
    if (!force_cpu) {
      new_op->set_engine("CUDNN");  // TODO: not here
    }
  }

  // copy externals
  if (firstModel.predict.net.op().size()) {
    // firstModel.predict.net.add_external_input(predict.Input(0));
  }
  if (secondModel.predict.net.op().size()) {
    // secondModel.predict.net.add_external_input(layer);
  }
  for (const auto &output : init.net.external_output()) {
    auto is_first = (static_inputs.find(output) != static_inputs.end());
    if (is_first) {
      firstModel.init.net.add_external_output(output);
    } else {
      secondModel.init.net.add_external_output(output);
    }
  }
  for (const auto &input : predict.net.external_input()) {
    auto is_first = (static_inputs.find(input) != static_inputs.end());
    if (is_first) {
      firstModel.predict.net.add_external_input(input);
    } else {
      secondModel.predict.net.add_external_input(input);
    }
  }
  if (firstModel.predict.net.op().size()) {
    firstModel.predict.net.add_external_output(layer);
  }
  if (secondModel.predict.net.op().size()) {
    secondModel.predict.net.add_external_output(predict.Output(0));
  }

  if (init.net.has_name()) {
    if (!firstModel.init.net.has_name()) {
      firstModel.init.SetName(init.net.name() + "_first");
    }
    if (!secondModel.init.net.has_name()) {
      secondModel.init.SetName(init.net.name() + "_second");
    }
  }
  if (predict.net.has_name()) {
    if (!firstModel.predict.net.has_name()) {
      firstModel.predict.SetName(predict.net.name() + "_first");
    }
    if (!secondModel.predict.net.has_name()) {
      secondModel.predict.SetName(predict.net.name() + "_second");
    }
  }
}

void set_trainable(OperatorDef &op, bool train) {
  if (op.type() == "Dropout" || op.type() == "SpatialBN") {
    for (auto &arg : *op.mutable_arg()) {
      if (arg.name() == "is_test") {
        arg.set_i(!train);
      }
    }
  }
}

void ModelUtil::CopyTrain(const std::string &layer, int out_size,
                          ModelUtil &train) const {
  std::string last_w, last_b;
  for (const auto &op : predict.net.op()) {
    auto new_op = train.predict.net.add_op();
    new_op->CopyFrom(op);
    set_trainable(*new_op, true);
    if (op.type() == "FC") {
      last_w = op.input(1);
      last_b = op.input(2);
    }
    if (op.type() == "SpatialBN") {
      if (op.output_size() < 2) new_op->add_output(op.input(3));
      if (op.output_size() < 3) new_op->add_output(op.input(4));
      if (op.output_size() < 4) new_op->add_output(op.input(3) + "_save");
      if (op.output_size() < 5) new_op->add_output(op.input(4) + "_save");
    }
  }
  train.predict.SetRenameInplace();
  for (const auto &op : init.net.op()) {
    auto &output = op.output(0);
    auto init_op = train.init.net.add_op();
    bool uniform = (output.find("_b") != std::string::npos);
    init_op->set_type(uniform ? "ConstantFill" : "XavierFill");
    for (const auto &arg : op.arg()) {
      if (arg.name() == "shape") {
        auto init_arg = init_op->add_arg();
        init_arg->set_name("shape");
        if (output == last_w) {
          init_arg->add_ints(out_size);
          init_arg->add_ints(arg.ints(1));
        } else if (output == last_b) {
          init_arg->add_ints(out_size);
        } else {
          init_arg->CopyFrom(arg);
        }
      }
    }
    init_op->add_output(output);
  }
  std::set<std::string> existing_inputs;
  existing_inputs.insert(train.predict.net.external_input().begin(),
                         train.predict.net.external_input().end());
  for (const auto &op : train.predict.net.op()) {
    auto inputs = op.input();
    for (auto &output : op.output()) {
      if (std::find(inputs.begin(), inputs.end(), output) == inputs.end()) {
        existing_inputs.insert(output);
      }
    }
  }
  for (const auto &input : predict.net.external_input()) {
    if (existing_inputs.find(input) == existing_inputs.end()) {
      train.predict.net.add_external_input(input);
    }
  }
  for (const auto &output : predict.net.external_output()) {
    train.predict.net.add_external_output(output);
  }
  // auto op = train_init_model.add_op();
  // op->set_type("ConstantFill");
  // auto arg = op->add_arg();
  // arg->set_name("shape");
  // arg->add_ints(1);
  // op->add_output(layer);
}

void ModelUtil::CopyTest(ModelUtil &test) const {
  for (const auto &op : predict.net.op()) {
    auto new_op = test.predict.net.add_op();
    new_op->CopyFrom(op);
    set_trainable(*new_op, false);
  }
  for (const auto &input : predict.net.external_input()) {
    test.predict.net.add_external_input(input);
  }
  for (const auto &output : predict.net.external_output()) {
    test.predict.net.add_external_output(output);
  }
}

void ModelUtil::CopyDeploy(ModelUtil &deploy, Workspace &workspace) const {
  for (const auto &op : init.net.op()) {
    auto &output = op.output(0);
    auto blob = workspace.GetBlob(output);
    if (blob) {
      auto tensor = BlobUtil(*blob).Get();
      auto init_op = deploy.init.net.add_op();
      init_op->set_type("GivenTensorFill");
      auto arg1 = init_op->add_arg();
      arg1->set_name("shape");
      for (auto dim : tensor.dims()) {
        arg1->add_ints(dim);
      }
      auto arg2 = init_op->add_arg();
      arg2->set_name("values");
      const auto &data = tensor.data<float>();
      for (auto i = 0; i < tensor.size(); ++i) {
        arg2->add_floats(data[i]);
      }
      init_op->add_output(output);
    } else {
      deploy.init.net.add_op()->CopyFrom(op);
    }
  }
}

size_t ModelUtil::Write(const std::string &path_prefix) const {
  size_t size = 0;
  size += init.Write(path_prefix + init_net_suffix);
  size += predict.Write(path_prefix + predict_net_suffix);
  return size;
}

size_t ModelUtil::Read(const std::string &path_prefix) {
  size_t size = 0;
  size += init.Read(path_prefix + init_net_suffix);
  size += predict.Read(path_prefix + predict_net_suffix);
  return size;
}

void ModelUtil::SetName(const std::string &name) {
  init.SetName(name + init_name_suffix);
  predict.SetName(name + predict_name_suffix);
}

void ModelUtil::SetDeviceCUDA() {
  init.SetDeviceCUDA();
  predict.SetDeviceCUDA();
}

std::string ModelUtil::Short() { return predict.Short() + init.Short(); }

}  // namespace caffe2
