#include "caffe2/core/init.h"
#include "caffe2/utils/proto_utils.h"
#include "caffe2/core/db.h"
#include "caffe2/core/operator_gradient.h"

#include "util/models.h"
#include "util/cuda.h"
#include "util/print.h"
#include "util/image.h"
#include "util/build.h"
#include "operator/operator_cout.h"
#include "res/imagenet_classes.h"

#include <dirent.h>
#include <sys/stat.h>

CAFFE2_DEFINE_string(model, "", "Name of one of the pre-trained models.");
CAFFE2_DEFINE_string(blob, "", "Name of the blob on which to split the model.");
CAFFE2_DEFINE_string(folder, "", "Folder with subfolders with images");

CAFFE2_DEFINE_string(db_type, "leveldb", "The database type.");
CAFFE2_DEFINE_int(size_to_fit, 224, "The image file.");
CAFFE2_DEFINE_int(image_mean, 128, "The mean to adjust values to.");
CAFFE2_DEFINE_int(train_runs, 100, "The of training runs.");
CAFFE2_DEFINE_int(test_runs, 50, "The of training runs.");
CAFFE2_DEFINE_int(batch_size, 64, "Training batch size.");
CAFFE2_DEFINE_double(learning_rate, 0.001, "Learning rate.");
CAFFE2_DEFINE_double(learning_gamma, 0.999, "Learning gamma.");
CAFFE2_DEFINE_bool(use_cuda, true, "Train on gpu if possible.");

enum {
  kRunTrain = 0,
  kRunValidate = 1,
  kRunTest = 2,
  kRunNum = 3
};

static std::map<int, std::string> name_for_run({
  { kRunTrain, "train" },
  { kRunValidate, "validate" },
  { kRunTest, "test" },
});

static std::map<int, int> percentage_for_run({
  { kRunTest, 10 },
  { kRunValidate, 20 },
  { kRunTrain, 70 },
});

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
  auto directory = opendir(FLAGS_folder.c_str());
  CHECK(directory) << "~ no image folder " << FLAGS_folder;
  if (directory) {
    struct stat s;
    struct dirent *entry;
    while ((entry = readdir(directory))) {
      auto class_name = entry->d_name;
      auto class_path = FLAGS_folder + '/' + class_name;
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
          closedir(subdir);
        }
      }
    }
    closedir(directory);
  }
  CHECK(image_files.size()) << "~ no images found in " << FLAGS_folder;
  std::random_shuffle(image_files.begin(), image_files.end());
  std::cout << class_labels.size() << " labels found" << std::endl;
  std::cout << image_files.size() << " images found" << std::endl;

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

void PreProcess(const std::vector<std::pair<std::string, int>> &image_files, const std::string *db_paths, NetDef &full_init_model, NetDef &pre_predict_model) {
  std::cout << "store partial prediction.." << std::endl;
  std::unique_ptr<db::DB> database[kRunNum];
  std::unique_ptr<db::Transaction> transaction[kRunNum];
  std::unique_ptr<db::Cursor> cursor[kRunNum];
  for (int i = 0; i < kRunNum; i++) {
    database[i] = db::CreateDB(FLAGS_db_type, db_paths[i], db::WRITE);
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
  Workspace workspace;
  auto init_net = CreateNet(full_init_model, &workspace);
  init_net->Run();
  auto predict_net = CreateNet(pre_predict_model, &workspace);
  auto input_name = pre_predict_model.external_input_size() ? pre_predict_model.external_input(0) : "";
  auto output_name = pre_predict_model.external_output_size() ? pre_predict_model.external_output(0) : "";
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
        std::cerr << '\r' << std::string(40, ' ') << '\r' << "pre-processing.. " << image_count << '/' << image_files.size() << " " << std::setprecision(3) << ((float)100 * image_count / image_files.size()) << "%" << std::flush;
        data->Clear();
        if (pre_predict_model.op_size()) {
          set_tensor_blob(*workspace.GetBlob(input_name), input);
          predict_net->Run();
          auto output = get_tensor_blob(*workspace.GetBlob(output_name));
          serializer.Serialize(output, "", data, 0, kDefaultChunkSize);
        } else {
          serializer.Serialize(input, "", data, 0, kDefaultChunkSize);
        }
        label->set_int32_data(0, class_index);
        protos.SerializeToString(&value);
        CHECK(value.size() < 65536) << "~ unfortunately caffe2's leveldb has block_size 65536, which is too small for our tensor data of size " << value.size();
        int percentage = 0;
        for (auto pair: percentage_for_run) {
          percentage += pair.second;
          if (image_count % 100 < percentage) {
            transaction[pair.first]->Put(filename, value);
            break;
          }
        }
        if (++insert_count % 100 == 0) {
          for (int i = 0; i < kRunNum && !in_db; i++) {
            transaction[i]->Commit();
          }
        }
        // std::cout << label->int32_data(0) << " " << value.size() << " " << output.size() << " " << filename << std::endl;
      }
    }
  }
  for (int i = 0; i < kRunNum; i++) {
    CHECK(database[i]->NewCursor()->Valid()) << "~ database " << name_for_run[i] << " is empty";
  }
  std::cerr << '\r' << std::string(80, ' ') << '\r' << image_files.size() << " images processed" << std::endl;
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

  if (!FLAGS_folder.size()) {
    std::cerr << "specify a image folder using --folder <name>" << std::endl;
    return;
  }

  if (!FLAGS_blob.size()) {
    std::cerr << "specify a layer blob using --blob <name>" << std::endl;
    return;
  }

  std::cout << "model: " << FLAGS_model << std::endl;
  std::cout << "blob_name: " << FLAGS_blob << std::endl;
  std::cout << "image_dir: " << FLAGS_folder << std::endl;
  std::cout << "db_type: " << FLAGS_db_type << std::endl;
  std::cout << "size_to_fit: " << FLAGS_size_to_fit << std::endl;
  std::cout << "image_mean: " << FLAGS_image_mean << std::endl;
  std::cout << "train_runs: " << FLAGS_train_runs << std::endl;
  std::cout << "test_runs: " << FLAGS_test_runs << std::endl;
  std::cout << "batch_size: " << FLAGS_batch_size << std::endl;
  std::cout << "learning_rate: " << FLAGS_learning_rate << std::endl;
  std::cout << "learning_gamma: " << FLAGS_learning_gamma << std::endl;
  std::cout << "use_cuda: " << FLAGS_use_cuda << std::endl;

  if(FLAGS_use_cuda && setupCUDA()) {
    std::cout << "using CUDA" << std::endl;
  }

  auto path_prefix = FLAGS_folder + '/' + '_' + FLAGS_model + '_' + FLAGS_blob + '_';
  std::string db_paths[kRunNum];
  for (int i = 0; i < kRunNum; i++) {
    db_paths[i] = path_prefix + name_for_run[i] + ".db";
  }

  std::cout << std::endl;

  auto load_time = -clock();
  std::vector<std::string> class_labels;
  std::vector<std::pair<std::string, int>> image_files;
  LoadLabels(path_prefix, class_labels, image_files);

  std::cout << "load model.." << std::endl;
  CHECK(ensureModel(FLAGS_model)) << "~ model " << FLAGS_model << " not found";
  NetDef full_init_model; // the original imagenet initialization model
  NetDef full_predict_model; // the original imagenet prediction model
  NetDef pre_predict_model; // predict model for pre-processing
  NetDef deploy_init_model; // the final initialization model
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
  deploy_init_model.set_name("retrain_" + full_init_model.name());
  pre_predict_model.set_name("pre_predict_model");
  if (FLAGS_use_cuda) {
    set_device_cuda_model(full_init_model);
    set_device_cuda_model(full_predict_model);
    set_device_cuda_model(pre_predict_model);
    for (int i = 0; i < kRunNum; i++) {
      set_device_cuda_model(init_model[i]);
      set_device_cuda_model(predict_model[i]);
    }
  }

  for (int i = 0; i < kRunNum; i++) {
    add_database_ops(init_model[i], predict_model[i], name_for_run[i], FLAGS_blob, db_paths[i], FLAGS_db_type, FLAGS_batch_size);
  }

  std::cout << "split model.." << std::endl;
  bool in_static = true;
  std::set<std::string> static_inputs;
  std::string last_w, last_b;
  std::vector<std::string> available_blobs;
  std::set<std::string> strip_op_types({ "Dropout" });
  for (const auto &op: full_predict_model.op()) {
    available_blobs.push_back(op.input(0));
    if (op.input(0) == FLAGS_blob && op.output(0) != FLAGS_blob) {
      in_static = false;
    }
    if (in_static) {
      auto new_op = pre_predict_model.add_op();
      new_op->CopyFrom(op);
      if (FLAGS_use_cuda) {
        set_engine_cudnn_op(*new_op);
      }
      for (const auto &input: op.input()) {
        static_inputs.insert(input);
      }
    } else {
      auto train_only = (strip_op_types.find(op.type()) != strip_op_types.end());
      for (int i = 0; i < (train_only ? 1 : kRunNum); i++) {
        auto new_op = predict_model[i].add_op();
        new_op->CopyFrom(op);
        if (FLAGS_use_cuda) {
          set_engine_cudnn_op(*new_op);
        }
      }
      if (op.type() == "FC") {
        last_w = op.input(1);
        last_b = op.input(2);
      }
    }
  }
  if (std::find(available_blobs.begin(), available_blobs.end(), FLAGS_blob) == available_blobs.end()) {
    std::cout << "available blobs:" << std::endl;
    available_blobs.erase(std::unique(available_blobs.begin(), available_blobs.end()), available_blobs.end());
    for (auto &blob: available_blobs) {
      std::cout << "  " << blob << std::endl;
    }
    LOG(FATAL) << "~ no blob with name " << FLAGS_blob << " in model.";
  }
  for (const auto &op: full_init_model.op()) {
    auto &output = op.output(0);
    if (static_inputs.find(output) == static_inputs.end()) {
      auto init_op = init_model[kRunTrain].add_op();
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
  auto op = init_model[kRunTrain].add_op();
  op->set_type("ConstantFill");
  auto arg = op->add_arg();
  arg->set_name("shape");
  arg->add_ints(1);
  op->add_output(FLAGS_blob);
  for (const auto &input: full_predict_model.external_input()) {
    if (static_inputs.find(input) != static_inputs.end()) {
      pre_predict_model.add_external_input(input);
    } else {
      predict_model[kRunTrain].add_external_input(input);
      predict_model[kRunValidate].add_external_input(input);
      predict_model[kRunTest].add_external_input(input);
    }
  }
  if (pre_predict_model.op().size()) {
    pre_predict_model.add_external_output(FLAGS_blob);
  }
  for (const auto &output: full_predict_model.external_output()) {
    for (int i = 0; i < kRunNum; i++) {
      predict_model[i].add_external_output(output);
    }
  }
  add_train_ops(init_model[kRunTrain], predict_model[kRunTrain], FLAGS_learning_rate, FLAGS_learning_gamma);
  add_test_ops(predict_model[kRunValidate]);
  add_test_ops(predict_model[kRunTest]);

  // std::cout << "full_init_model -------------" << std::endl;
  // print(full_init_model);
  // std::cout << "full_predict_model -------------" << std::endl;
  // print(full_predict_model);
  // std::cout << "pre_predict_model -------------" << std::endl;
  // print(pre_predict_model);
  // std::cout << "init_model[kRunTrain] -------------" << std::endl;
  // print(init_model[kRunTrain]);
  // std::cout << "predict_model[kRunTrain] -------------" << std::endl;
  // print(predict_model[kRunTrain]);

  PreProcess(image_files, db_paths, full_init_model, pre_predict_model);
  load_time += clock();


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

    if (i % 10 == 0) {
      auto iter = get_tensor_blob(*workspace.GetBlob("iter")).data<int64_t>()[0];
      auto lr = get_tensor_blob(*workspace.GetBlob("LR")).data<float>()[0];
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

  for (const auto &op: full_init_model.op()) {
    auto &output = op.output(0);
    if (static_inputs.find(output) == static_inputs.end()) {
      auto tensor = get_tensor_blob(*workspace.GetBlob(output));
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
