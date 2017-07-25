#include "util/misc.h"

CAFFE2_DEFINE_string(path, "res/mnist-test-nchw-leveldb", "path of the database");
CAFFE2_DEFINE_string(db_type, "leveldb", "The database type.");

namespace caffe2 {

void run() {
  std::cout << std::endl;
  std::cout << "## Database inspector ##" << std::endl;
  std::cout << std::endl;

  std::cout << "path: " << FLAGS_path << std::endl;
  std::cout << "db_type: " << FLAGS_db_type << std::endl;

  dump_database(FLAGS_path, FLAGS_db_type);
}

}  // namespace caffe2

int main(int argc, char** argv) {
  caffe2::GlobalInit(&argc, &argv);
  caffe2::run();
  google::protobuf::ShutdownProtobufLibrary();
  return 0;
}
