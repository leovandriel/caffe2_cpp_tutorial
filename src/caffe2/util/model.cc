#include "model.pb.h"

#include "caffe2/util/blob.h"
#include "caffe2/util/model.h"

#include <fcntl.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

namespace caffe2 {

const std::string gradient_suffix("_grad");
const std::string moment_suffix("_moment");
const std::string meansq_suffix("_meansq");
const std::string reader_suffix("_reader");
const std::string init_net_suffix("_init_net.pb");
const std::string predict_net_suffix("_predict_net.pb");
const std::string model_info_suffix("_model_info.pb");
const std::string init_name_suffix("_init");
const std::string predict_name_suffix("_predict");

const std::string iter_name("iter");
const std::string lr_name("lr");
const std::string one_name("one");
const std::string loss_name("loss");
const std::string label_name("label");
const std::string xent_name("xent");
const std::string accuracy_name("accuracy");

ModelUtil::ModelUtil(NetDef &init_net, NetDef &predict_net,
                     const std::string &name)
    : init(init_net), predict(predict_net) {
  if (name.size()) {
    SetName(name);
  }
  meta = new ModelMeta();
}

ModelUtil::ModelUtil(NetUtil &init, NetUtil &predict)
    : init(init), predict(predict) {
  meta = new ModelMeta();
}

ModelUtil::~ModelUtil() { delete meta; }

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

void ModelUtil::AddTrainOps(const std::string &output, float base_rate,
                            std::string &optimizer) {
  AddXentOps(output);
  predict.AddConstantFillWithOp(1.0, loss_name, loss_name + gradient_suffix);
  predict.AddGradientOps();
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
    predict.AddSpatialBNOp({input, output + "_scale", output + "_bias",
                            output + "_mean", output + "_var"},
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
  if (op.type() == "SpatialBN") {
    if (train) {
      if (op.output_size() < 2) op.add_output(op.input(3));
      if (op.output_size() < 3) op.add_output(op.input(4));
      if (op.output_size() < 4) op.add_output(op.input(3) + "_save");
      if (op.output_size() < 5) op.add_output(op.input(4) + "_save");
    } else if (op.output_size() > 1) {
      auto output = op.output(0);
      op.clear_output();
      op.add_output(output);
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
  }
  train.predict.SetRenameInplace();
  for (const auto &op : init.net.op()) {
    auto &output = op.output(0);
    auto init_op = train.init.net.add_op();
    init_op->set_type(op.type());
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
      } else if (arg.name() == "value") {
        auto init_arg = init_op->add_arg();
        init_arg->set_name("value");
        init_arg->CopyFrom(arg);
      }
    }
    init_op->add_output(output);
  }
  train.init.SetFillToTrain();
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

void WriteProto(const std::string &filename, const MessageLite &message,
                bool optional = false) {
  int fd = open(filename.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (!optional) {
    CAFFE_ENFORCE_NE(fd, -1, "File cannot be created: ", filename,
                     " error number: ", errno);
  } else if (fd == -1) {
    return;
  }
  std::unique_ptr<::google::protobuf::io::ZeroCopyOutputStream> raw_output(
      new ::google::protobuf::io::FileOutputStream(fd));
  std::unique_ptr<::google::protobuf::io::CodedOutputStream> coded_output(
      new ::google::protobuf::io::CodedOutputStream(raw_output.get()));
  CAFFE_ENFORCE(message.SerializeToCodedStream(coded_output.get()),
                "Unable to write protobuf message");
  coded_output.reset();
  raw_output.reset();
  close(fd);
}

void ReadProto(const std::string &filename, MessageLite &message,
               bool optional = false) {
  int fd = open(filename.c_str(), O_RDONLY);
  if (!optional) {
    CAFFE_ENFORCE_NE(fd, -1, "File not found: ", filename);
  } else if (fd == -1) {
    return;
  }
  std::unique_ptr<::google::protobuf::io::ZeroCopyInputStream> raw_input(
      new ::google::protobuf::io::FileInputStream(fd));
  std::unique_ptr<::google::protobuf::io::CodedInputStream> coded_input(
      new ::google::protobuf::io::CodedInputStream(raw_input.get()));
  coded_input->SetTotalBytesLimit(1024LL << 20, 512LL << 20);
  CAFFE_ENFORCE(message.ParseFromCodedStream(coded_input.get()),
                "Unable to read protobuf message");
  coded_input.reset();
  raw_input.reset();
  close(fd);
}

size_t ModelUtil::WriteMeta(const std::string &filename) const {
  WriteProto(filename, *meta, true);
  return std::ifstream(filename, std::ifstream::ate | std::ifstream::binary)
      .tellg();
}

size_t ModelUtil::ReadMeta(const std::string &filename) {
  ReadProto(filename, *meta, true);
  return std::ifstream(filename, std::ifstream::ate | std::ifstream::binary)
      .tellg();
}

size_t ModelUtil::Write(const std::string &path_prefix) const {
  size_t size = 0;
  size += init.Write(path_prefix + init_net_suffix);
  size += predict.Write(path_prefix + predict_net_suffix);
  size += WriteMeta(path_prefix + model_info_suffix);
  return size;
}

size_t ModelUtil::Read(const std::string &path_prefix) {
  size_t size = 0;
  size += init.Read(path_prefix + init_net_suffix);
  size += predict.Read(path_prefix + predict_net_suffix);
  size += ReadMeta(path_prefix + model_info_suffix);
  return size;
}

size_t ModelUtil::WriteBundle(const std::string &filename) const {
  ModelDef model;
  ModelMeta i = *meta;
  model.set_allocated_meta(&i);
  model.set_allocated_predict(&predict.net);
  model.set_allocated_init(&init.net);
  WriteProto(filename, model);
  model.release_meta();
  model.release_predict();
  model.release_init();
  return std::ifstream(filename, std::ifstream::ate | std::ifstream::binary)
      .tellg();
}

size_t ModelUtil::ReadBundle(const std::string &filename) {
  ModelDef model;
  model.set_allocated_meta(meta);
  model.set_allocated_predict(&predict.net);
  model.set_allocated_init(&init.net);
  ReadProto(filename, model);
  model.release_meta();
  model.release_predict();
  model.release_init();
  return std::ifstream(filename, std::ifstream::ate | std::ifstream::binary)
      .tellg();
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
std::string ModelUtil::Proto() { return predict.Proto() + init.Proto(); }

void ModelUtil::input_dims(const std::vector<int> &dims) {
  auto input = meta->mutable_input();
  input->clear_dims();
  for (auto d : dims) {
    input->add_dims(d);
  }
}

std::vector<int> ModelUtil::input_dims() {
  auto &dims = meta->input().dims();
  return std::vector<int>(dims.begin(), dims.end());
}

void ModelUtil::output_labels(const std::vector<std::string> &labels) {
  auto output = meta->mutable_output();
  output->clear_labels();
  for (auto l : labels) {
    output->add_labels(l);
  }
}

std::vector<std::string> ModelUtil::output_labels() {
  auto &labels = meta->output().labels();
  return std::vector<std::string>(labels.begin(), labels.end());
}

}  // namespace caffe2
