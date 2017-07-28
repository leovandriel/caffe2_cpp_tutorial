#include "caffe2/core/init.h"
#include "caffe2/core/net.h"
#include "caffe2/zoo/keeper.h"

#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "google/protobuf/text_format.h"

CAFFE2_DEFINE_string(model, "googlenet", "Name of model.");
CAFFE2_DEFINE_bool(short, false, "use short format.");

CAFFE2_DEFINE_bool(code, false, "use in-code model.");
CAFFE2_DEFINE_bool(file, false, "use in-file model.");

namespace caffe2 {

void run() {
  NetDef init_model, predict_model;

  if (FLAGS_code && !FLAGS_file) {
    Keeper(FLAGS_model).AddModel(init_model, predict_model, false);
  } else if (!FLAGS_code && FLAGS_file) {
    Keeper(FLAGS_model).AddModel(init_model, predict_model, true);
    NetUtil(init_model).SetFillToTrain();
  } else {
    std::cerr << "set either --code or --file" << std::endl;
  }

  if (FLAGS_short) {
    std::cout << NetUtil(init_model).Short();
    std::cout << NetUtil(predict_model).Short();
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
