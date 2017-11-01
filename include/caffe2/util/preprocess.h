#ifndef PREPROCESS_H
#define PREPROCESS_H

#include <caffe2/core/db.h>
#include <caffe2/core/init.h>
#include <caffe2/core/net.h>

#include "caffe2/util/blob.h"
#include "caffe2/util/model.h"
#include "caffe2/util/net.h"
#include "caffe2/util/tensor.h"
#include "caffe2/util/train.h"

#include <dirent.h>
#include <sys/stat.h>

namespace caffe2 {

static std::map<int, int> percentage_for_run({
    {kRunTest, 10}, {kRunValidate, 20}, {kRunTrain, 70},
});

std::string filename_to_key(const std::string &filename) {
  // return filename;
  return std::to_string(std::hash<std::string>{}(filename)) + "_" + filename;
}

void load_labels(const std::string &folder, const std::string &path_prefix,
                 std::vector<std::string> &class_labels,
                 std::vector<std::pair<std::string, int>> &image_files) {
  auto classes_text_path = path_prefix + "classes.txt";
  ;
  std::ifstream infile(classes_text_path);
  std::string line;
  while (std::getline(infile, line)) {
    if (line.size()) {
      class_labels.push_back(line);
    }
  }

  auto directory = opendir(folder.c_str());
  CAFFE_ENFORCE(directory, "no image folder " + folder);
  if (directory) {
    struct stat s;
    struct dirent *entry;
    while ((entry = readdir(directory))) {
      auto class_name = entry->d_name;
      auto class_path = folder + '/' + class_name;
      if (class_name[0] != '.' && class_name[0] != '_' &&
          !stat(class_path.c_str(), &s) && (s.st_mode & S_IFDIR)) {
        auto subdir = opendir(class_path.c_str());
        if (subdir) {
          auto class_index =
              find(class_labels.begin(), class_labels.end(), class_name) -
              class_labels.begin();
          if (class_index == class_labels.size()) {
            class_labels.push_back(class_name);
          }
          while ((entry = readdir(subdir))) {
            auto image_file = entry->d_name;
            auto image_path = class_path + '/' + image_file;
            if (image_file[0] != '.' && !stat(image_path.c_str(), &s) &&
                (s.st_mode & S_IFREG)) {
              image_files.push_back({image_path, class_index});
            }
          }
          closedir(subdir);
        }
      }
    }
    closedir(directory);
  }
  CAFFE_ENFORCE(image_files.size(), "no images found in " + folder);
  std::random_shuffle(image_files.begin(), image_files.end());

  std::ofstream class_file(classes_text_path);
  if (class_file.is_open()) {
    for (auto &label : class_labels) {
      class_file << label << std::endl;
    }
    class_file.close();
  }
  auto classes_header_path = path_prefix + "classes.h";
  std::ofstream labels_file(classes_header_path.c_str());
  if (labels_file.is_open()) {
    labels_file << "const char * retrain_classes[] {";
    bool first = true;
    for (auto &label : class_labels) {
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

int write_batch(Workspace &workspace, ModelUtil &model, std::string &input_name,
                std::string &output_name,
                std::vector<std::pair<std::string, int>> &batch_files,
                std::unique_ptr<db::DB> *database, int size_to_fit) {
  std::unique_ptr<db::Transaction> transaction[kRunNum];
  for (int i = 0; i < kRunNum; i++) {
    transaction[i] = database[i]->NewTransaction();
  }

  std::vector<std::string> filenames;
  for (auto &pair : batch_files) {
    filenames.push_back(pair.first);
  }
  std::vector<int> indices;
  TensorCPU input;
  TensorUtil(input).ReadImages(filenames, size_to_fit, indices);
  TensorCPU output;
  if (model.predict.net.external_input_size() && input.size() > 0) {
    BlobUtil(*workspace.GetBlob(input_name)).Set(input);
    CAFFE_ENFORCE(workspace.RunNet(model.predict.net.name()));
    auto tensor = BlobUtil(*workspace.GetBlob(output_name)).Get();
    output.ResizeLike(tensor);
    output.ShareData(tensor);
  } else {
    output.ResizeLike(input);
    output.ShareData(input);
  }

  TensorProtos protos;
  TensorProto *data = protos.add_protos();
  TensorProto *label = protos.add_protos();
  data->set_data_type(TensorProto::FLOAT);
  label->set_data_type(TensorProto::INT32);
  label->add_int32_data(0);
  TensorSerializer<CPUContext> serializer;
  std::string value;
  std::vector<TIndex> dims(output.dims().begin() + 1, output.dims().end());
  auto size = output.dim(0) ? output.size() / output.dim(0) : 0;
  auto output_data = output.data<float>();
  for (auto i : indices) {
    auto single = TensorCPU(
        dims, std::vector<float>(output_data, output_data + size), NULL);
    output_data += size;
    data->Clear();
    serializer.Serialize(single, "", data, 0, kDefaultChunkSize);
    label->set_int32_data(0, batch_files[i].second);
    protos.SerializeToString(&value);
    int percentage = 0, p = (int)(rand() * 100.0 / RAND_MAX);
    auto key = filename_to_key(batch_files[i].first);
    for (auto pair : percentage_for_run) {
      percentage += pair.second;
      if (p < percentage) {
        transaction[pair.first]->Put(key, value);
        break;
      }
    }
  }

  for (int i = 0; i < kRunNum; i++) {
    transaction[i]->Commit();
  }

  return indices.size();
}

int preprocess(const std::vector<std::pair<std::string, int>> &image_files,
               const std::string *db_paths, ModelUtil &model,
               const std::string &db_type, int batch_size, int size_to_fit) {
  std::unique_ptr<db::DB> database[kRunNum];
  for (int i = 0; i < kRunNum; i++) {
    database[i] = db::CreateDB(db_type, db_paths[i], db::WRITE);
  }
  auto image_count = 0;
  auto sample_count = 0;
  Workspace workspace;
  CAFFE_ENFORCE(workspace.RunNetOnce(model.init.net));
  if (model.predict.net.external_input_size()) {
    CAFFE_ENFORCE(workspace.CreateNet(model.predict.net));
  }
  auto input_name = model.predict.net.external_input_size()
                        ? model.predict.net.external_input(0)
                        : "";
  auto output_name = model.predict.net.external_output_size()
                         ? model.predict.net.external_output(0)
                         : "";
  std::vector<std::pair<std::string, int>> batch_files;
  for (auto &pair : image_files) {
    auto &filename = pair.first;
    auto class_index = pair.second;
    image_count++;
    auto in_db = false;
    auto key = filename_to_key(filename);
    for (int i = 0; i < kRunNum && !in_db; i++) {
      auto cursor = database[i]->NewCursor();
      cursor->Seek(key);
      in_db |= (cursor->Valid() && cursor->key() == key);
    }
    if (in_db) {
      sample_count++;
    } else {
      batch_files.push_back({filename, class_index});
    }
    if (image_count % 10 == 0) {
      std::cerr << '\r' << std::string(40, ' ') << '\r' << "pre-processing.. "
                << image_count << '/' << image_files.size() << " "
                << std::setprecision(3)
                << ((float)100 * image_count / image_files.size()) << "%"
                << std::flush;
    }
    if (batch_files.size() == batch_size) {
      sample_count += write_batch(workspace, model, input_name, output_name,
                                  batch_files, database, size_to_fit);
      batch_files.clear();
    }
  }
  if (batch_files.size() > 0) {
    sample_count += write_batch(workspace, model, input_name, output_name,
                                batch_files, database, size_to_fit);
  }
  for (int i = 0; i < kRunNum; i++) {
    CAFFE_ENFORCE(database[i]->NewCursor()->Valid(),
                  "database " + name_for_run[i] + " is empty");
  }
  std::cerr << '\r' << std::string(80, ' ') << '\r';

  return sample_count;
}

void preprocess(const std::vector<std::pair<std::string, int>> &image_files,
                const std::string *db_paths, const std::string &db_type,
                int size_to_fit) {
  NetDef n;
  ModelUtil none(n, n);
  preprocess(image_files, db_paths, none, db_type, 64, size_to_fit);
}

int count_samples(const std::string *db_paths, const std::string &db_type) {
  std::unique_ptr<db::DB> database[kRunNum];
  for (int i = 0; i < kRunNum; i++) {
    database[i] = db::CreateDB(db_type, db_paths[i], db::WRITE);
  }
  auto sample_count = 0;
  for (int i = 0; i < kRunNum; i++) {
    auto cursor = database[i]->NewCursor();
    while (cursor->Valid()) {
      sample_count++;
      cursor->Next();
    }
  }
  return sample_count;
}

}  // namespace caffe2

#endif  // PREPROCESS_H
