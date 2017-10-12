#include <caffe2/core/db.h>
#include <caffe2/core/init.h>

#include "caffe2/util/model.h"
#include "caffe2/util/net.h"
#include "caffe2/util/tensor.h"

CAFFE2_DEFINE_string(path, "res/mnist-test-nchw-leveldb",
                     "path of the database");
CAFFE2_DEFINE_string(db_type, "leveldb", "The database type.");

namespace caffe2 {

void dump_database(const std::string db_path, const std::string& db_type) {
  std::cout << "dumping database.." << std::endl;
  std::unique_ptr<db::DB> database = db::CreateDB(db_type, db_path, db::READ);

  for (auto cursor = database->NewCursor(); cursor->Valid(); cursor->Next()) {
    auto key = cursor->key().substr(0, 48);
    auto value = cursor->value();
    TensorProtos protos;
    protos.ParseFromString(value);
    auto tensor_proto = protos.protos(0);
    auto label_proto = protos.protos(1);
    TensorDeserializer<CPUContext> deserializer;
    TensorCPU tensor;
    int label = label_proto.int32_data(0);
    deserializer.Deserialize(tensor_proto, &tensor);
    auto dims = tensor.dims();
    dims.insert(dims.begin(), 1);
    tensor.Resize(dims);
    std::cout << key << "  "
              << (value.size() > 1000 ? value.size() / 1000 : value.size())
              << (value.size() > 1000 ? "K" : "B") << "  (" << tensor.dims()
              << ")  " << label << std::endl;
    TensorUtil(tensor).ShowImage("inspect", 0, 1.0, 128);
  }
}

void run() {
  std::cout << std::endl;
  std::cout << "## Database inspector ##" << std::endl;
  std::cout << std::endl;

  std::cout << "path: " << FLAGS_path << std::endl;
  std::cout << "db-type: " << FLAGS_db_type << std::endl;

  dump_database(FLAGS_path, FLAGS_db_type);
}

}  // namespace caffe2

int main(int argc, char** argv) {
  caffe2::GlobalInit(&argc, &argv);
  caffe2::run();
  google::protobuf::ShutdownProtobufLibrary();
  return 0;
}
