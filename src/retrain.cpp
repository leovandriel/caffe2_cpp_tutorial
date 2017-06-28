#include "caffe2/core/init.h"
#include "caffe2/core/predictor.h"
#include "caffe2/utils/proto_utils.h"
#include "caffe2/core/db.h"
#include "caffe2/core/operator_gradient.h"

#include "util/models.h"
#include "util/print.h"
#include "util/image.h"
#include "operator/operator_cout.h"
#include "res/imagenet_classes.h"

#include <dirent.h>
#include <sys/stat.h>

CAFFE2_DEFINE_string(model, "alexnet", "Name of one of the pre-trained models.");
CAFFE2_DEFINE_string(blob_name, "fc6", "Name of the blob on which to split the model.");
CAFFE2_DEFINE_string(image_dir, "retrain", "Folder with subfolders with images");
CAFFE2_DEFINE_string(db_type, "leveldb", "The database type.");
CAFFE2_DEFINE_int(size_to_fit, 224, "The image file.");
CAFFE2_DEFINE_int(image_mean, 128, "The mean to adjust values to.");
CAFFE2_DEFINE_int(train_runs, 50, "The of training runs.");
CAFFE2_DEFINE_int(test_runs, 50, "The of training runs.");

enum {
  kRunTrain = 0,
  kRunValidate = 1,
  kRunTest = 2,
  kRunNum = 3
};

namespace caffe2 {

void LoadLabels(const std::string &path_prefix, std::vector<std::string> &class_labels, std::vector<std::pair<std::string, int>> &image_files) {
  std::cout << "load class labels.." << std::endl;
  auto classes_text_path = path_prefix + "classes.txt";
  ;
  std::ifstream infile(classes_text_path);
  std::string line;
  while (std::getline(infile, line)) {
    if (line.size()) {
      class_labels.push_back(line);
      // std::cout << '.' << line << '.' << std::endl;
    }
  }

  std::cout << "load image folder.." << std::endl;
  auto directory = opendir(FLAGS_image_dir.c_str());
  CAFFE_ENFORCE(directory);
  if (directory) {
    struct stat s;
    struct dirent *entry;
    while ((entry = readdir(directory))) {
      auto class_name = entry->d_name;
      auto class_path = FLAGS_image_dir + '/' + class_name;
      if (class_name[0] != '.' && class_name[0] != '_' && !stat(class_path.c_str(), &s) && (s.st_mode & S_IFDIR)) {
        auto subdir = opendir(class_path.c_str());
        if (subdir) {
          auto class_index = find(class_labels.begin(), class_labels.end(), class_name) - class_labels.begin();
          if (class_index == class_labels.size()) {
            class_labels.push_back(class_name);
          }
          while ((entry = readdir(subdir))) {
            auto image_file = entry->d_name;
            auto image_path = class_path + '/' + image_file;
            if (image_file[0] != '.' && !stat(image_path.c_str(), &s) && (s.st_mode & S_IFREG)) {
              // std::cout << class_name << ' ' <<  image_path << std::endl;
              image_files.push_back({ image_path, class_index });
            }
          }
        }
        closedir(subdir);
      }
    }
    closedir(directory);
  }

  std::cout << class_labels.size() << " labels found" << std::endl;

  std::cout << "write class labels.." << std::endl;
  std::ofstream class_file(classes_text_path);
  if (class_file.is_open()) {
    for (auto &label: class_labels) {
      class_file << label << std::endl;
    }
    class_file.close();
  }
  auto classes_header_path = path_prefix + "classes.h";
  std::ofstream labels_file(classes_header_path.c_str());
  if (labels_file.is_open()) {
    labels_file << "const char * retrain_classes[] {";
    bool first = true;
    for (auto &label: class_labels) {
      if (first) {
        first = false;
      } else {
        labels_file << ',';
      }
      labels_file << std::endl << '"' << label << '"';
    }
    labels_file << std::endl << "};" << std::endl;
    labels_file.close();
  }
}

void PreProcess(const std::vector<std::pair<std::string, int>> &image_files, const std::string *run_paths, NetDef &full_init_model, NetDef &static_predict_model) {
  std::cout << "store partial prediction.." << std::endl;
  std::unique_ptr<db::DB> database[kRunNum];
  std::unique_ptr<db::Transaction> transaction[kRunNum];
  std::unique_ptr<db::Cursor> cursor[kRunNum];
  for (int i = 0; i < kRunNum; i++) {
    database[i] = db::CreateDB(FLAGS_db_type, run_paths[i], db::WRITE);
    transaction[i] = database[i]->NewTransaction();
    cursor[i] = database[i]->NewCursor();
  }
  TensorProtos protos;
  TensorProto* data = protos.add_protos();
  TensorProto* label = protos.add_protos();
  data->set_data_type(TensorProto::FLOAT);
  label->set_data_type(TensorProto::INT32);
  label->add_int32_data(0);
  auto insert_count = 0;
  auto image_count = 0;
  std::string value;
  Predictor predictor(full_init_model, static_predict_model);
  TensorSerializer<CPUContext> serializer;
  for (auto &pair: image_files) {
    auto &filename = pair.first;
    auto class_index = pair.second;
    image_count++;
    auto in_db = false;
    for (int i = 0; i < kRunNum && !in_db; i++) {
      cursor[i]->Seek(filename);
      in_db |= (cursor[i]->Valid() && cursor[i]->key() == filename);
    }
    if (!in_db) {
      auto input = readImageTensor(filename, FLAGS_size_to_fit, -FLAGS_image_mean);
      if (input.size() > 0) {
        std::cerr << '\r' << std::string(32, ' ') << '\r' << "pre-processing.. " << std::setprecision(3) << ((float)100 * image_count / image_files.size()) << "%" << std::flush;
        Predictor::TensorVector inputVec({ &input }), outputVec;
        predictor.run(inputVec, &outputVec);
        auto &output = *(outputVec[0]);
        label->set_int32_data(0, class_index);
        data->Clear();
        serializer.Serialize(output, "", data, 0, kDefaultChunkSize);
        protos.SerializeToString(&value);
        if (image_count % 10 < 1) {
          transaction[kRunTest]->Put(filename, value);
        } else if (image_count % 10 < 3) {
          transaction[kRunValidate]->Put(filename, value);
        } else {
          transaction[kRunTrain]->Put(filename, value);
        }
        if (++insert_count % 1000 == 0) {
          for (int i = 0; i < kRunNum && !in_db; i++) {
            transaction[i]->Commit();
          }
        }
        // std::cout << label->int32_data(0) << " " << value.size() << " " << output.size() << " " << filename << std::endl;
      }
    }
  }
  std::cerr << '\r' << std::string(80, ' ') << '\r' << image_files.size() << " images processed" << std::endl;
}

void AddInitOps(NetDef &init_model, NetDef &predict_model, const std::string &db, const std::string &db_type, int batch_size) {
  {
    auto op = init_model.add_op();
    op->set_type("CreateDB");
    auto arg1 = op->add_arg();
    arg1->set_name("db_type");
    arg1->set_s(db_type);
    auto arg2 = op->add_arg();
    arg2->set_name("db");
    arg2->set_s(db);
    op->add_output("dbreader");
    predict_model.add_external_input("dbreader");
  }
  {
    auto op = predict_model.add_op();
    op->set_type("TensorProtosDBInput");
    auto arg = op->add_arg();
    arg->set_name("batch_size");
    arg->set_i(batch_size);
    op->add_input("dbreader");
    op->add_output(FLAGS_blob_name);
    op->add_output("label");
  }
  {
    auto op = init_model.add_op();
    op->set_type("ConstantFill");
    auto arg1 = op->add_arg();
    arg1->set_name("shape");
    arg1->add_ints(1);
    auto arg2 = op->add_arg();
    arg2->set_name("value");
    arg2->set_i(0);
    auto arg3 = op->add_arg();
    arg3->set_name("dtype");
    arg3->set_i(TensorProto_DataType_INT64);
    op->add_output("ITER");
    predict_model.add_external_input("ITER");
  }
}

void AddAccuracyOps(NetDef &predict_model) {
  {
    auto op = predict_model.add_op();
    op->set_type("Accuracy");
    op->add_input("prob");
    op->add_input("label");
    op->add_output("accuracy");
  }
  {
    auto op = predict_model.add_op();
    op->set_type("Iter");
    op->add_input("ITER");
    op->add_output("ITER");
  }
}

void AddTrainingOps(NetDef &init_model, NetDef &predict_model) {
  std::vector<std::string> params;
  std::vector<const OperatorDef *> gradient_ops;
  for (const auto &op: predict_model.op()) {
    if (op.type() == "FC") {
      params.push_back(op.input(1));
      params.push_back(op.input(2));
      gradient_ops.push_back(&op);
    } else if (op.type() == "Relu" || op.type() == "Softmax") {
      gradient_ops.push_back(&op);
    }
  }

  {
    auto op = predict_model.add_op();
    op->set_type("LabelCrossEntropy");
    op->add_input("prob");
    op->add_input("label");
    op->add_output("xent");
    gradient_ops.push_back(op);
  }
  {
    auto op = predict_model.add_op();
    op->set_type("AveragedLoss");
    op->add_input("xent");
    op->add_output("loss");
    gradient_ops.push_back(op);
  }

  AddAccuracyOps(predict_model);

  {
    auto op = predict_model.add_op();
    op->set_type("ConstantFill");
    auto arg = op->add_arg();
    arg->set_name("value");
    arg->set_f(1.0);
    op->add_input("loss");
    op->add_output("loss_grad");
    op->set_is_gradient_op(true);
  }
  std::reverse(gradient_ops.begin(), gradient_ops.end());
  for (auto op: gradient_ops) {
    vector<GradientWrapper> output(op->output_size());
    for (auto i = 0; i < output.size(); i++) {
      output[i].dense_ = op->output(i) + "_grad";
    }
    GradientOpsMeta meta = GetGradientForOp(*op, output);
    auto grad = predict_model.add_op();
    grad->CopyFrom(meta.ops_[0]);
    grad->set_is_gradient_op(true);
  }

  {
    auto op = predict_model.add_op();
    op->set_type("LearningRate");
    auto arg1 = op->add_arg();
    arg1->set_name("policy");
    arg1->set_s("step");
    auto arg2 = op->add_arg();
    arg2->set_name("stepsize");
    arg2->set_i(1);
    auto arg3 = op->add_arg();
    arg3->set_name("base_lr");
    arg3->set_f(-0.01);
    auto arg4 = op->add_arg();
    arg4->set_name("gamma");
    arg4->set_f(0.999);
    op->add_input("ITER");
    op->add_output("LR");
  }
  {
    auto op = init_model.add_op();
    op->set_type("ConstantFill");
    auto arg1 = op->add_arg();
    arg1->set_name("shape");
    arg1->add_ints(1);
    auto arg2 = op->add_arg();
    arg2->set_name("value");
    arg2->set_f(1.0);
    op->add_output("ONE");
    predict_model.add_external_input("ONE");
  }

  for (auto param: params) {
    auto op = predict_model.add_op();
    op->set_type("WeightedSum");
    op->add_input(param);
    op->add_input("ONE");
    op->add_input(param + "_grad");
    op->add_input("LR");
    op->add_output(param);
  }
}

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

  std::cout << "model: " << FLAGS_model << std::endl;
  std::cout << "blob_name: " << FLAGS_blob_name << std::endl;
  std::cout << "image_dir: " << FLAGS_image_dir << std::endl;
  std::cout << "size_to_fit: " << FLAGS_size_to_fit << std::endl;
  std::cout << "image_mean: " << FLAGS_image_mean << std::endl;
  auto path_prefix = FLAGS_image_dir + '/' + '_' + FLAGS_model + '_' + FLAGS_blob_name + '_';

  std::string run_paths[kRunNum];
  run_paths[kRunTrain] = path_prefix + "train.db";
  run_paths[kRunValidate] = path_prefix + "validate.db";
  run_paths[kRunTest] = path_prefix + "test.db";

  std::cout << std::endl;

  std::vector<std::string> class_labels;
  std::vector<std::pair<std::string, int>> image_files;
  LoadLabels(path_prefix, class_labels, image_files);

  std::cout << "load model.." << std::endl;
  CAFFE_ENFORCE(ensureModel(FLAGS_model), "model ", FLAGS_model, " not found");
  NetDef full_init_model; // the original imagenet initialization model
  NetDef full_predict_model; // the original imagenet prediction model
  NetDef static_predict_model; // predict model for pre-processing
  NetDef dynamic_init_model; // init model for training
  NetDef dynamic_train_model; // train predict model for training
  NetDef dynamic_test_model; // test predict model for training
  NetDef deploy_init_model; // the final initialization model
  std::string init_filename = "res/" + FLAGS_model + "_init_net.pb";
  std::string predict_filename = "res/" + FLAGS_model + "_predict_net.pb";
  CAFFE_ENFORCE(ReadProtoFromFile(init_filename.c_str(), &full_init_model));
  CAFFE_ENFORCE(ReadProtoFromFile(predict_filename.c_str(), &full_predict_model));
  static_predict_model.set_name("static_predict_model");
  dynamic_init_model.set_name("dynamic_init_model");
  dynamic_train_model.set_name("dynamic_train_model");
  dynamic_test_model.set_name("dynamic_test_model");

  AddInitOps(dynamic_init_model, dynamic_train_model, run_paths[kRunTrain], FLAGS_db_type, 64);
  AddInitOps(dynamic_init_model, dynamic_test_model, run_paths[kRunValidate], FLAGS_db_type, 64);

  std::cout << "split model.." << std::endl;
  bool in_static = true;
  std::set<std::string> static_inputs;
  std::string last_w, last_b;
  for (const auto &op: full_predict_model.op()) {
    if (op.input(0) == FLAGS_blob_name && op.output(0) != FLAGS_blob_name) {
      in_static = false;
    }
    if (in_static) {
      static_predict_model.add_op()->CopyFrom(op);
      for (const auto &input: op.input()) {
        static_inputs.insert(input);
      }
    } else {
      dynamic_train_model.add_op()->CopyFrom(op);
      if (op.type() != "Dropout") {
        dynamic_test_model.add_op()->CopyFrom(op);
      }
      if (op.type() == "FC") {
        last_w = op.input(1);
        last_b = op.input(2);
      }
    }
  }
  for (const auto &op: full_init_model.op()) {
    auto &output = op.output(0);
    if (static_inputs.find(output) == static_inputs.end()) {
      auto init_op = dynamic_init_model.add_op();
      bool uniform = (output.find("_b") != std::string::npos);
      init_op->set_type(uniform ? "ConstantFill" : "XavierFill");
      for (const auto &arg: op.arg()) {
        if (arg.name() == "shape") {
          auto init_arg = init_op->add_arg();
          init_arg->set_name("shape");
          if (output == last_w) {
            init_arg->add_ints(class_labels.size());
            init_arg->add_ints(arg.ints(1));
          } else if (output == last_b) {
            init_arg->add_ints(class_labels.size());
          } else {
            init_arg->CopyFrom(arg);
          }
        }
      }
      init_op->add_output(output);
    }
  }
  auto op = dynamic_init_model.add_op();
  op->set_type("ConstantFill");
  auto arg = op->add_arg();
  arg->set_name("shape");
  arg->add_ints(1);
  op->add_output(FLAGS_blob_name);
  for (const auto &input: full_predict_model.external_input()) {
    if (static_inputs.find(input) != static_inputs.end()) {
      static_predict_model.add_external_input(input);
    } else {
      dynamic_train_model.add_external_input(input);
      dynamic_test_model.add_external_input(input);
    }
  }
  static_predict_model.add_external_output(FLAGS_blob_name);
  for (const auto &output: full_predict_model.external_output()) {
    // dynamic_train_model.add_external_output(output);
  }
  dynamic_train_model.add_external_output(full_predict_model.external_output(0));
  dynamic_test_model.add_external_output(full_predict_model.external_output(0));
  AddTrainingOps(dynamic_init_model, dynamic_train_model);
  AddAccuracyOps(dynamic_test_model);

  // std::cout << "full_init_model -------------" << std::endl;
  // print(full_init_model);
  // std::cout << "full_predict_model -------------" << std::endl;
  // print(full_predict_model);
  // std::cout << "static_predict_model -------------" << std::endl;
  // print(static_predict_model);
  Workspace workspace("tmp");

  PreProcess(image_files, run_paths, full_init_model, static_predict_model);

  // std::cout << "dynamic_init_model -------------" << std::endl;
  // print(dynamic_init_model);
  // std::cout << "dynamic_train_model -------------" << std::endl;
  // print(dynamic_train_model);
  // std::cout << "dynamic_test_model -------------" << std::endl;
  // print(dynamic_test_model);

  std::cout << "training.." << std::endl;
  auto dynamic_init_net = CreateNet(dynamic_init_model, &workspace);
  dynamic_init_net->Run();
  auto dynamic_train_net = CreateNet(dynamic_train_model, &workspace);
  for (auto i = 1; i <= FLAGS_train_runs; i++) {
    dynamic_train_net->Run();

    if (i % 10 == 0) {
      auto accuracy = workspace.GetBlob("accuracy")->Get<TensorCPU>().data<float>()[0];
      auto loss = workspace.GetBlob("loss")->Get<TensorCPU>().data<float>()[0];
      auto iter = workspace.GetBlob("ITER")->Get<TensorCPU>().data<long long>()[0];
      auto lr = workspace.GetBlob("LR")->Get<TensorCPU>().data<float>()[0];
      std::cout << "step: " << iter << " rate: " << lr << " loss: " << loss << " accuracy: " << accuracy << std::endl;
    }
  }

  std::cout << "testing.." << std::endl;
  auto dynamic_test_net = CreateNet(dynamic_test_model, &workspace);
  for (auto i = 1; i <= FLAGS_test_runs; i++) {
    dynamic_test_net->Run();

    if (i % 10 == 0) {
      auto accuracy = workspace.GetBlob("accuracy")->Get<TensorCPU>().data<float>()[0];
      auto loss = workspace.GetBlob("loss")->Get<TensorCPU>().data<float>()[0];
      auto iter = workspace.GetBlob("ITER")->Get<TensorCPU>().data<long long>()[0];
      std::cout << "step: " << iter << " loss: " << loss << " accuracy: " << accuracy << std::endl;
    }
  }

  for (const auto &op: full_init_model.op()) {
    auto &output = op.output(0);
    if (static_inputs.find(output) == static_inputs.end()) {
      auto tensor = workspace.GetBlob(output)->Get<TensorCPU>();
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

  // std::cout << "full_init_model -------------" << std::endl;
  // print(full_init_model);
  // std::cout << "deploy_init_model -------------" << std::endl;
  // print(deploy_init_model);


  WriteProtoToBinaryFile(deploy_init_model, path_prefix + "init_net.pb");
  WriteProtoToBinaryFile(full_predict_model, path_prefix + "predict_net.pb");

  // load model

  // run net
  // Predictor predictor(init_model, predict_model);
  // Predictor::TensorVector inputVec({ &input }), outputVec;
  // auto predict_time = clock();
  // predictor.run(inputVec, &outputVec);
  // auto finish_time = clock();

  // printBest(*(outputVec[0]), imagenet_classes);

  std::cout << std::endl;

  // std::cout << std::setprecision(3) << "load: " << ((float)(predict_time - read_time) / CLOCKS_PER_SEC) << "s  predict: " << ((float)(finish_time - predict_time) / CLOCKS_PER_SEC) << "s  model: " << ((float)model_size / 1000000) << "MB" << std::endl;
}

}  // namespace caffe2

int main(int argc, char** argv) {
  caffe2::GlobalInit(&argc, &argv);
  caffe2::run();
  google::protobuf::ShutdownProtobufLibrary();
  return 0;
}
