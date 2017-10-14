#include "caffe2/util/train.h"
#include <caffe2/core/db.h>
#include <caffe2/core/init.h>
#include <caffe2/core/operator_gradient.h>
#include <caffe2/utils/proto_utils.h>
#include "caffe2/util/plot.h"
#include "caffe2/util/preprocess.h"
#include "caffe2/util/window.h"
#include "caffe2/zoo/keeper.h"

#include "res/imagenet_classes.h"

CAFFE2_DEFINE_string(model, "", "Name of one of the pre-trained models.");
CAFFE2_DEFINE_string(layer, "",
                     "Name of the layer on which to split the model.");
CAFFE2_DEFINE_string(folder, "", "Folder with subfolders with images");

CAFFE2_DEFINE_string(db_type, "leveldb", "The database type.");
CAFFE2_DEFINE_int(size_to_fit, 224, "The image file.");
CAFFE2_DEFINE_int(train_runs, 1000, "The of training runs.");
CAFFE2_DEFINE_int(test_runs, 50, "The of training runs.");
CAFFE2_DEFINE_int(batch_size, 64, "Training batch size.");
CAFFE2_DEFINE_double(learning_rate, 1e-4, "Learning rate.");

CAFFE2_DEFINE_bool(zero_one, false, "Show zero-one for batch.");
CAFFE2_DEFINE_bool(display, false,
                   "Show worst correct and incorrect classification.");
CAFFE2_DEFINE_bool(reshape_output, false,
                   "Reshape output (necessary for squeeznet)");

#include "caffe2/util/cmd.h"

namespace caffe2 {

void run() {
  if (!cmd_init("CNN Training Example")) {
    return;
  }

  if (!FLAGS_model.size()) {
    std::cerr << "specify a model name using --model <name>" << std::endl;
    for (auto const &pair : keeper_model_lookup) {
      std::cerr << "  " << pair.first << std::endl;
    }
    return;
  }

  if (!FLAGS_folder.size()) {
    std::cerr << "specify a image folder using --folder <name>" << std::endl;
    return;
  }

  std::cout << "model: " << FLAGS_model << std::endl;
  std::cout << "layer: " << FLAGS_layer << std::endl;
  std::cout << "image-dir: " << FLAGS_folder << std::endl;
  std::cout << "db-type: " << FLAGS_db_type << std::endl;
  std::cout << "size-to-fit: " << FLAGS_size_to_fit << std::endl;
  std::cout << "train-runs: " << FLAGS_train_runs << std::endl;
  std::cout << "test-runs: " << FLAGS_test_runs << std::endl;
  std::cout << "batch-size: " << FLAGS_batch_size << std::endl;
  std::cout << "learning-rate: " << FLAGS_learning_rate << std::endl;
  std::cout << "zero-one: " << (FLAGS_zero_one ? "true" : "false") << std::endl;
  std::cout << "display: " << (FLAGS_display ? "true" : "false") << std::endl;
  std::cout << "reshape-output: " << (FLAGS_reshape_output ? "true" : "false")
            << std::endl;

  auto has_split = FLAGS_layer.size() > 0;
  std::string layer_prefix;
  std::string model_safe = FLAGS_model;
  std::replace(model_safe.begin(), model_safe.end(), '/', '_');
  if (has_split) {
    std::string layer_safe = FLAGS_layer;
    std::replace(layer_safe.begin(), layer_safe.end(), '/', '_');
    std::replace(layer_safe.begin(), layer_safe.end(), '.', '_');
    layer_prefix = layer_safe + '_';
  }
  auto path_prefix = FLAGS_folder + '/' + '_' + layer_prefix;

  if (FLAGS_display) {
    superWindow("Full Train Example");
    if (!has_split) {
      moveWindow("undercertain", 0, 0);
      resizeWindow("undercertain", 300, 300);
      setWindowTitle("undercertain", "uncertain but correct");
      moveWindow("overcertain", 0, 300);
      resizeWindow("overcertain", 300, 300);
      setWindowTitle("overcertain", "certain but incorrect");
    }
    moveWindow("accuracy", has_split ? 0 : 300, 0);
    resizeWindow("accuracy", 500, 300);
    moveWindow("loss", has_split ? 0 : 300, 300);
    resizeWindow("loss", 500, 300);
  }

  std::string db_paths[kRunNum];
  for (int i = 0; i < kRunNum; i++) {
    db_paths[i] = path_prefix + name_for_run[i] + ".db";
  }

  std::cout << std::endl;

  std::cout << "collect images.." << std::endl;
  auto load_time = -clock();
  std::vector<std::string> class_labels;
  std::vector<std::pair<std::string, int>> image_files;
  load_labels(FLAGS_folder, path_prefix, class_labels, image_files);
  std::cout << class_labels.size() << " labels found" << std::endl;
  std::cout << image_files.size() << " images found" << std::endl;

  std::cout << "load model.." << std::endl;
  NetDef full_init_model, full_predict_model;
  ModelUtil full(full_init_model, full_predict_model);
  Keeper(FLAGS_model).AddModel(full, has_split, class_labels.size());

  if (FLAGS_device == "cudnn") {
    full.init.SetEngineOps("CUDNN");
    full.predict.SetEngineOps("CUDNN");
  }

  NetDef init_model[kRunNum], predict_model[kRunNum];
  ModelUtil models[kRunNum] = {
      {init_model[kRunTrain], predict_model[kRunTrain],
       name_for_run[kRunTrain]},
      {init_model[kRunTest], predict_model[kRunTest], name_for_run[kRunTest]},
      {init_model[kRunValidate], predict_model[kRunValidate],
       name_for_run[kRunValidate]},
  };

  NetDef first_init_model, first_predict_model;
  ModelUtil first(first_init_model, first_predict_model);
  NetDef second_init_model, second_predict_model;
  ModelUtil second(second_init_model, second_predict_model);

  if (has_split) {
    full.predict.CheckLayerAvailable(FLAGS_layer);
    std::cout << "split model.. (at " << FLAGS_layer << ")" << std::endl;
    full.Split(FLAGS_layer, first, second, FLAGS_device != "cudnn");
    if (FLAGS_device != "cpu") {
      first.SetDeviceCUDA();
    }
  } else {
    second.init.net = full.init.net;
    second.predict.net = full.predict.net;
  }

  std::cout << "cache images.." << std::endl;
  auto count = preprocess(image_files, db_paths, first, FLAGS_db_type,
                          FLAGS_batch_size, FLAGS_size_to_fit);
  std::cout << count << " images processed" << std::endl;
  load_time += clock();

  auto model_in = has_split ? FLAGS_layer : full.predict.Input(0);
  for (int i = 0; i < kRunNum; i++) {
    models[i].AddDatabaseOps(name_for_run[i], model_in, db_paths[i],
                             FLAGS_db_type, FLAGS_batch_size);
  }
  second.CopyTrain(model_in, class_labels.size(), models[kRunTrain]);
  second.CopyTest(models[kRunValidate]);
  second.CopyTest(models[kRunTest]);

  auto output = models[kRunTrain].predict.Output(0);
  if (FLAGS_reshape_output) {
    auto output_reshaped = output + "_reshaped";
    for (int i = 0; i < kRunNum; i++) {
      models[i].predict.AddReshapeOp(output, output_reshaped, {0, -1});
    }
    output = output_reshaped;
  }

  models[kRunTrain].AddTrainOps(output, FLAGS_learning_rate, FLAGS_optimizer);
  ModelUtil(second.predict, models[kRunValidate].predict).AddTestOps(output);
  ModelUtil(second.predict, models[kRunTest].predict).AddTestOps(output);

  if (FLAGS_zero_one) {
    models[kRunValidate].predict.AddZeroOneOp(output, "label");
  }

  if (FLAGS_display) {
    if (!has_split) {
      models[kRunValidate].predict.AddShowWorstOp(output, "label",
                                                  second.predict.Input(0));
    }
    models[kRunTrain].predict.AddTimePlotOp("accuracy", "iter", "accuracy",
                                            "train", 10);
    models[kRunValidate].predict.AddTimePlotOp("accuracy", "iter", "accuracy",
                                               "test");
    models[kRunTrain].predict.AddTimePlotOp("loss", "iter", "loss", "train",
                                            10);
    models[kRunValidate].predict.AddTimePlotOp("loss", "iter", "loss", "test");
  }

  if (FLAGS_device != "cpu") {
    for (int i = 0; i < kRunNum; i++) {
      models[i].SetDeviceCUDA();
    }
  }

  if (FLAGS_dump_model) {
    std::cout << models[kRunTrain].Short();
  }

  std::cout << std::endl;

  Workspace workspace("tmp");

  clock_t train_time = 0;
  clock_t validate_time = 0;
  clock_t test_time = 0;

  std::cout << "training.." << std::endl;
  run_trainer(FLAGS_train_runs, models[kRunTrain], models[kRunValidate],
              workspace, train_time, validate_time);

  std::cout << std::endl;
  std::cout << "testing.." << std::endl;
  run_tester(FLAGS_test_runs, models[kRunTest], workspace, test_time);

  NetDef deploy_init_model;  // the final initialization model
  ModelUtil deploy(deploy_init_model, full.predict.net,
                   "train_" + full.init.net.name());
  full.CopyDeploy(deploy, workspace);

  size_t model_size = deploy.Write(path_prefix + model_safe + '_');

  std::cout << std::endl;

  std::cout << std::setprecision(3)
            << "load: " << ((float)load_time / CLOCKS_PER_SEC)
            << "s  train: " << ((float)train_time / CLOCKS_PER_SEC)
            << "s  validate: " << ((float)validate_time / CLOCKS_PER_SEC)
            << "s  test: " << ((float)test_time / CLOCKS_PER_SEC)
            << "s  model: " << ((float)model_size / 1000000) << "MB"
            << std::endl;
}

}  // namespace caffe2

int main(int argc, char **argv) {
  caffe2::GlobalInit(&argc, &argv);
  caffe2::run();
  google::protobuf::ShutdownProtobufLibrary();
  return 0;
}
