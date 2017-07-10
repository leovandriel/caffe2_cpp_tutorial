#include "util/net.h"

#include "caffe2/core/init.h"
#include "caffe2/utils/proto_utils.h"
#include "caffe2/core/db.h"
#include "caffe2/core/operator_gradient.h"

#include "util/models.h"
#include "util/cuda.h"
#include "util/print.h"
#include "util/image.h"
#include "util/build.h"
#include "operator/cout_op.h"
#include "res/imagenet_classes.h"

CAFFE2_DEFINE_string(model, "", "Name of one of the pre-trained models.");
CAFFE2_DEFINE_string(layer, "", "Name of the layer on which to split the model.");
CAFFE2_DEFINE_string(folder, "", "Folder with subfolders with images");

CAFFE2_DEFINE_string(db_type, "leveldb", "The database type.");
CAFFE2_DEFINE_int(size_to_fit, 224, "The image file.");
CAFFE2_DEFINE_int(train_runs, 100 * caffe2::cuda_multipier, "The of training runs.");
CAFFE2_DEFINE_int(test_runs, 50, "The of training runs.");
CAFFE2_DEFINE_int(batch_size, 64, "Training batch size.");
CAFFE2_DEFINE_double(learning_rate, 1e-4, "Learning rate.");
CAFFE2_DEFINE_string(optimizer, "adam", "Training optimizer: sgd/momentum/adagrad/adam");
CAFFE2_DEFINE_bool(force_cpu, false, "Only use CPU, no CUDA.");

namespace caffe2 {

void run() {
  std::cout << std::endl;
  std::cout << "## Partial Retrain Example ##" << std::endl;
  std::cout << std::endl;

  if (!FLAGS_model.size()) {
    std::cerr << "specify a model name using --model <name>" << std::endl;
    for (auto const &pair: model_lookup) {
      std::cerr << "  " << pair.first << std::endl;
    }
    return;
  }

  if (!FLAGS_folder.size()) {
    std::cerr << "specify a image folder using --folder <name>" << std::endl;
    return;
  }

  if (!FLAGS_layer.size()) {
    std::cerr << "specify a layer layer using --layer <name>" << std::endl;
    return;
  }

  std::cout << "model: " << FLAGS_model << std::endl;
  std::cout << "layer: " << FLAGS_layer << std::endl;
  std::cout << "image_dir: " << FLAGS_folder << std::endl;
  std::cout << "db_type: " << FLAGS_db_type << std::endl;
  std::cout << "size_to_fit: " << FLAGS_size_to_fit << std::endl;
  std::cout << "train_runs: " << FLAGS_train_runs << std::endl;
  std::cout << "test_runs: " << FLAGS_test_runs << std::endl;
  std::cout << "batch_size: " << FLAGS_batch_size << std::endl;
  std::cout << "learning_rate: " << FLAGS_learning_rate << std::endl;
  std::cout << "optimizer: " << FLAGS_optimizer << std::endl;
  std::cout << "force_cpu: " << (FLAGS_force_cpu ? "true" : "false") << std::endl;

  if (!FLAGS_force_cpu) setupCUDA();

  std::string layer_safe = FLAGS_layer;
  std::string::size_type index = 0;
  while ((index = layer_safe.find("/", index)) != std::string::npos) {
    layer_safe.replace(index, 3, "_");
    index += 1;
  }

  auto path_prefix = FLAGS_folder + '/' + '_' + FLAGS_model + '_' + layer_safe + '_';
  std::string db_paths[kRunNum];
  for (int i = 0; i < kRunNum; i++) {
    db_paths[i] = path_prefix + name_for_run[i] + ".db";
  }

  std::cout << std::endl;

  auto load_time = -clock();
  std::vector<std::string> class_labels;
  std::vector<std::pair<std::string, int>> image_files;
  LoadLabels(FLAGS_folder, path_prefix, class_labels, image_files);

  std::cout << "load model.." << std::endl;
  CHECK(ensureModel(FLAGS_model)) << "~ model " << FLAGS_model << " not found";
  NetDef full_init_model; // the original imagenet initialization model
  NetDef full_predict_model; // the original imagenet prediction model
  NetDef init_model[kRunNum];
  NetDef predict_model[kRunNum];
  for (int i = 0; i < kRunNum; i++) {
    init_model[i].set_name(name_for_run[i] + "_init_model");
    predict_model[i].set_name(name_for_run[i] + "_predict_model");
  }
  std::string init_filename = "res/" + FLAGS_model + "_init_net.pb";
  std::string predict_filename = "res/" + FLAGS_model + "_predict_net.pb";
  CHECK(ReadProtoFromFile(init_filename.c_str(), &full_init_model)) << "~ empty init model " << init_filename;
  CHECK(ReadProtoFromFile(predict_filename.c_str(), &full_predict_model)) << "~ empty predict model " << predict_filename;

  CheckLayerAvailable(full_predict_model, FLAGS_layer);

  // std::cout << join_net(full_init_model);
  // std::cout << join_net(full_predict_model);

  NetDef first_init_model, first_predict_model, second_init_model, second_predict_model;
  SplitModel(full_init_model, full_predict_model, FLAGS_layer, first_init_model, first_predict_model, second_init_model, second_predict_model, FLAGS_force_cpu);

  if (!FLAGS_force_cpu) {
    set_device_cuda_model(first_init_model);
    set_device_cuda_model(first_predict_model);
  }

  PreProcess(image_files, db_paths, first_init_model, first_predict_model, FLAGS_db_type, FLAGS_batch_size, FLAGS_size_to_fit);
  load_time += clock();

  // std::cout << join_net(first_init_model);
  // std::cout << join_net(first_predict_model);
  // std::cout << join_net(second_init_model);
  // std::cout << join_net(second_predict_model);

  for (int i = 0; i < kRunNum; i++) {
    add_database_ops(init_model[i], predict_model[i], name_for_run[i], FLAGS_layer, db_paths[i], FLAGS_db_type, FLAGS_batch_size);
  }
  TrainModel(second_init_model, second_predict_model, FLAGS_layer, class_labels.size(), init_model[kRunTrain], predict_model[kRunTrain], FLAGS_learning_rate, FLAGS_optimizer);
  TestModel(second_predict_model, predict_model[kRunValidate]);
  TestModel(second_predict_model, predict_model[kRunTest]);

  if (!FLAGS_force_cpu) {
    for (int i = 0; i < kRunNum; i++) {
      set_device_cuda_model(init_model[i]);
      set_device_cuda_model(predict_model[i]);
    }
  }

  // std::cout << join_net(init_model[kRunTrain]);
  // std::cout << join_net(predict_model[kRunTrain]);


  // std::cout << join_net(init_model[kRunValidate]);
  // std::cout << join_net(init_model[kRunTest]);

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

  std::cout << "training.." << std::endl;
  for (auto i = 1; i <= FLAGS_train_runs; i++) {
    train_time -= clock();
    predict_net[kRunTrain]->Run();
    train_time += clock();

    if (i % (10 * cuda_multipier) == 0) {
      auto iter = get_tensor_blob(*workspace.GetBlob("iter")).data<int64_t>()[0];
      auto lr = get_tensor_blob(*workspace.GetBlob("lr")).data<float>()[0];
      auto train_accuracy = get_tensor_blob(*workspace.GetBlob("accuracy")).data<float>()[0];
      auto train_loss = get_tensor_blob(*workspace.GetBlob("loss")).data<float>()[0];
      validate_time -= clock();
      predict_net[kRunValidate]->Run();
      validate_time += clock();
      auto validate_accuracy = get_tensor_blob(*workspace.GetBlob("accuracy")).data<float>()[0];
      auto validate_loss = get_tensor_blob(*workspace.GetBlob("loss")).data<float>()[0];
      std::cout << "step: " << iter << "  rate: " << lr << "  loss: " << train_loss << " | " << validate_loss << "  accuracy: " << train_accuracy << " | " << validate_accuracy << std::endl;
    }
  }

  std::cout << std::endl;

  std::cout << "testing.." << std::endl;
  for (auto i = 1; i <= FLAGS_test_runs; i++) {
    test_time -= clock();
    predict_net[kRunTest]->Run();
    test_time += clock();

    if (i % 10 == 0) {
      auto accuracy = get_tensor_blob(*workspace.GetBlob("accuracy")).data<float>()[0];
      auto loss = get_tensor_blob(*workspace.GetBlob("loss")).data<float>()[0];
      std::cout << "step: " << i << " loss: " << loss << " accuracy: " << accuracy << std::endl;
    }
  }

  NetDef deploy_init_model; // the final initialization model
  deploy_init_model.set_name("retrain_" + full_init_model.name());
  for (const auto &op: full_init_model.op()) {
    auto &output = op.output(0);
    auto blob = workspace.GetBlob(output);
    if (blob) {
      auto tensor = get_tensor_blob(*blob);
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

  // std::cout << join_net(full_init_model);
  // std::cout << join_net(deploy_init_model);

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
