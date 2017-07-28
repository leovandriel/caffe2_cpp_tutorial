#include "util/misc.h"

#include "caffe2/core/init.h"
#include "caffe2/utils/proto_utils.h"
#include "caffe2/core/db.h"
#include "caffe2/core/operator_gradient.h"
#include "caffe2/zoo/keeper.h"

#include "res/imagenet_classes.h"

CAFFE2_DEFINE_string(model, "", "Name of one of the pre-trained models.");
CAFFE2_DEFINE_string(folder, "", "Folder with subfolders with images");

CAFFE2_DEFINE_string(db_type, "leveldb", "The database type.");
CAFFE2_DEFINE_int(size_to_fit, 224, "The image file.");
CAFFE2_DEFINE_int(train_runs, 1000, "The of training runs.");
CAFFE2_DEFINE_int(test_runs, 50, "The of training runs.");
CAFFE2_DEFINE_int(batch_size, 64, "Training batch size.");
CAFFE2_DEFINE_double(learning_rate, 1e-4, "Learning rate.");

CAFFE2_DEFINE_bool(zero_one, false, "Show zero-one for batch.");
CAFFE2_DEFINE_bool(show_worst, false, "Show worst correct and incorrect classification.");

#include "util/cmd.h"

namespace caffe2 {

void run() {
  if (!cmd_init("Full Train Example")) {
    return;
  }

  if (!FLAGS_model.size()) {
    std::cerr << "specify a model name using --model <name>" << std::endl;
    for (auto const &pair: keeper_model_lookup) {
      std::cerr << "  " << pair.first << std::endl;
    }
    return;
  }

  if (!FLAGS_folder.size()) {
    std::cerr << "specify a image folder using --folder <name>" << std::endl;
    return;
  }

  std::cout << "model: " << FLAGS_model << std::endl;
  std::cout << "image_dir: " << FLAGS_folder << std::endl;
  std::cout << "db_type: " << FLAGS_db_type << std::endl;
  std::cout << "size_to_fit: " << FLAGS_size_to_fit << std::endl;
  std::cout << "train_runs: " << FLAGS_train_runs << std::endl;
  std::cout << "test_runs: " << FLAGS_test_runs << std::endl;
  std::cout << "batch_size: " << FLAGS_batch_size << std::endl;
  std::cout << "learning_rate: " << FLAGS_learning_rate << std::endl;
  std::cout << "zero_one: " << (FLAGS_zero_one ? "true" : "false") << std::endl;
  std::cout << "show_worst: " << (FLAGS_show_worst ? "true" : "false") << std::endl;

  auto path_prefix = FLAGS_folder + '/' + '_' + FLAGS_model + '_';
  std::string db_paths[kRunNum];
  for (int i = 0; i < kRunNum; i++) {
    db_paths[i] = path_prefix + name_for_run[i] + ".db";
  }

  std::cout << std::endl;

  auto load_time = -clock();
  std::vector<std::string> class_labels;
  std::vector<std::pair<std::string, int>> image_files;
  load_labels(FLAGS_folder, path_prefix, class_labels, image_files);

  std::cout << "load model.." << std::endl;
  NetDef full_init_model, full_predict_model;
  Keeper(FLAGS_model).AddModel(full_init_model, full_predict_model, false);

  if (FLAGS_device == "cudnn") {
    NetUtil(full_init_model).SetEngineOps("CUDNN");
    NetUtil(full_predict_model).SetEngineOps("CUDNN");
  }

  if (FLAGS_dump_model) {
    std::cout << NetUtil(full_init_model).Short();
    std::cout << NetUtil(full_predict_model).Short();
  }

  NetDef init_model[kRunNum];
  NetDef predict_model[kRunNum];
  for (int i = 0; i < kRunNum; i++) {
    init_model[i].set_name(name_for_run[i] + "_init_model");
    predict_model[i].set_name(name_for_run[i] + "_predict_model");
  }

  pre_process(image_files, db_paths, FLAGS_db_type, FLAGS_size_to_fit);
  load_time += clock();

  for (int i = 0; i < kRunNum; i++) {
    ModelUtil(init_model[i], predict_model[i]).AddDatabaseOps(name_for_run[i], full_predict_model.external_input(0), db_paths[i], FLAGS_db_type, FLAGS_batch_size);
  }
  add_train_model(full_init_model, full_predict_model, full_predict_model.external_input(0), class_labels.size(), init_model[kRunTrain], predict_model[kRunTrain], FLAGS_learning_rate, FLAGS_optimizer);
  add_test_model(full_predict_model, predict_model[kRunValidate]);
  add_test_model(full_predict_model, predict_model[kRunTest]);
  if (FLAGS_zero_one) {
    NetUtil(predict_model[kRunValidate]).AddZeroOneOp(full_predict_model.external_output(0), "label");
  }
  if (FLAGS_show_worst) {
    NetUtil(predict_model[kRunValidate]).AddShowWorstOp(full_predict_model.external_output(0), "label", full_predict_model.external_input(0));
  }

  if (FLAGS_device != "cpu") {
    for (int i = 0; i < kRunNum; i++) {
      NetUtil(init_model[i]).SetDeviceCUDA();
      NetUtil(predict_model[i]).SetDeviceCUDA();
    }
  }

  std::cout << std::endl;

  Workspace workspace("tmp");
  unique_ptr<caffe2::NetBase> predict_net[kRunNum];
  for (int i = 0; i < kRunNum; i++) {
    auto init_net = CreateNet(init_model[i], &workspace);
    init_net->Run();
    predict_net[i] = CreateNet(predict_model[i], &workspace);
  }

  clock_t train_time = 0;
  clock_t validate_time = 0;
  clock_t test_time = 0;

  auto last_time = clock();
  auto last_i = 0;

  std::cout << "training.." << std::endl;
  for (auto i = 1; i <= FLAGS_train_runs; i++) {
    train_time -= clock();
    predict_net[kRunTrain]->Run();
    train_time += clock();

    auto steps_time = (float)(clock() - last_time) / CLOCKS_PER_SEC;
    if (steps_time > 5) {
      auto iter = BlobUtil(*workspace.GetBlob("iter")).Get().data<int64_t>()[0];
      auto lr = BlobUtil(*workspace.GetBlob("lr")).Get().data<float>()[0];
      auto train_accuracy = BlobUtil(*workspace.GetBlob("accuracy")).Get().data<float>()[0];
      auto train_loss = BlobUtil(*workspace.GetBlob("loss")).Get().data<float>()[0];
      validate_time -= clock();
      predict_net[kRunValidate]->Run();
      validate_time += clock();
      auto validate_accuracy = BlobUtil(*workspace.GetBlob("accuracy")).Get().data<float>()[0];
      auto validate_loss = BlobUtil(*workspace.GetBlob("loss")).Get().data<float>()[0];
      std::cout << "step: " << iter << "  rate: " << lr << "  loss: " << train_loss << " | " << validate_loss << "  accuracy: " << train_accuracy << " | " << validate_accuracy << "  step_time: " << std::setprecision(3) << steps_time / (i - last_i) << "s" << std::endl;
      last_i = i;
      last_time = clock();
    }
  }

  std::cout << std::endl;

  std::cout << "testing.." << std::endl;
  for (auto i = 1; i <= FLAGS_test_runs; i++) {
    test_time -= clock();
    predict_net[kRunTest]->Run();
    test_time += clock();

    if (i % 10 == 0) {
      auto accuracy = BlobUtil(*workspace.GetBlob("accuracy")).Get().data<float>()[0];
      auto loss = BlobUtil(*workspace.GetBlob("loss")).Get().data<float>()[0];
      std::cout << "step: " << i << " loss: " << loss << " accuracy: " << accuracy << std::endl;
    }
  }

  NetDef deploy_init_model; // the final initialization model
  deploy_init_model.set_name("train_" + full_init_model.name());
  for (const auto &op: full_init_model.op()) {
    auto &output = op.output(0);
    auto blob = workspace.GetBlob(output);
    if (blob) {
      auto tensor = BlobUtil(*blob).Get();
      auto init_op = deploy_init_model.add_op();
      init_op->set_type("GivenTensorFill");
      auto arg1 = init_op->add_arg();
      arg1->set_name("shape");
      for (auto dim: tensor.dims()) {
        arg1->add_ints(dim);
      }
      auto arg2 = init_op->add_arg();
      arg2->set_name("values");
      const auto& data = tensor.data<float>();
      for (auto i = 0; i < tensor.size(); ++i) {
        arg2->add_floats(data[i]);
      }
      init_op->add_output(output);
    } else {
      deploy_init_model.add_op()->CopyFrom(op);
    }
  }

  WriteProtoToBinaryFile(deploy_init_model, path_prefix + "init_net.pb");
  WriteProtoToBinaryFile(full_predict_model, path_prefix + "predict_net.pb");
  auto init_size = std::ifstream(path_prefix + "init_net.pb", std::ifstream::ate | std::ifstream::binary).tellg();
  auto predict_size = std::ifstream(path_prefix + "predict_net.pb", std::ifstream::ate | std::ifstream::binary).tellg();
  auto model_size = init_size + predict_size;

  std::cout << std::endl;

  std::cout << std::setprecision(3) << "load: " << ((float)load_time / CLOCKS_PER_SEC) << "s  train: " << ((float)train_time / CLOCKS_PER_SEC) << "s  validate: " << ((float)validate_time / CLOCKS_PER_SEC) << "s  test: " << ((float)test_time / CLOCKS_PER_SEC) << "s  model: " << ((float)model_size / 1000000) << "MB" << std::endl;
}

}  // namespace caffe2

int main(int argc, char** argv) {
  caffe2::GlobalInit(&argc, &argv);
  caffe2::run();
  google::protobuf::ShutdownProtobufLibrary();
  return 0;
}
