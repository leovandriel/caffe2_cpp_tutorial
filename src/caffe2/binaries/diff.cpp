#include "caffe2/core/init.h"
#include "caffe2/core/net.h"

#include "util/zoo.h"
#include "util/print.h"

#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "google/protobuf/text_format.h"

CAFFE2_DEFINE_bool(short, false, "use short format.");

CAFFE2_DEFINE_bool(code, false, "use in-code model.");
CAFFE2_DEFINE_bool(file, false, "use in-file model.");

namespace caffe2 {

void run() {
  NetDef init_model, predict_model;

  if (FLAGS_code && !FLAGS_file) {
    std::string model = "googlenet";
    add_model(model, init_model, predict_model);
  } else if (!FLAGS_code && FLAGS_file) {
    std::string model = "googlenet";
    ReadProtoFromFile(("res/" + model + "_init_net.pb").c_str(), &init_model);
    ReadProtoFromFile(("res/" + model + "_predict_net.pb").c_str(), &predict_model);
    NetUtil(init_model).SetFillToTrain();
  } else {
    std::cerr << "set either --code or --file" << std::endl;
  }

  if (FLAGS_short) {
    std::cout << join_net(init_model);
    std::cout << join_net(predict_model);
  } else {
    google::protobuf::io::OstreamOutputStream stream(&std::cout);
    google::protobuf::TextFormat::Print(init_model, &stream);
    google::protobuf::TextFormat::Print(predict_model, &stream);
  }
}

}  // namespace caffe2

int main(int argc, char** argv) {
  caffe2::GlobalInit(&argc, &argv);
  caffe2::run();
  google::protobuf::ShutdownProtobufLibrary();
  return 0;
}
