#include <caffe2/core/init.h>
#include <caffe2/core/net.h>
#include "caffe2/zoo/keeper.h"

#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

CAFFE2_DEFINE_string(model, "googlenet", "Name of model.");
CAFFE2_DEFINE_bool(short, false, "use short format.");

CAFFE2_DEFINE_bool(code, false, "use in-code model.");
CAFFE2_DEFINE_bool(file, false, "use in-file model.");

namespace caffe2 {

void run() {
  NetDef init_model, predict_model;
  ModelUtil model(init_model, predict_model);

  if (FLAGS_code && !FLAGS_file) {
    Keeper(FLAGS_model).AddModel(model, false);
  } else if (!FLAGS_code && FLAGS_file) {
    Keeper(FLAGS_model).AddModel(model, true);
    model.init.SetFillToTrain();
  } else {
    std::cerr << "set either --code or --file" << std::endl;
  }

  if (FLAGS_short) {
    std::cout << model.Short();
  } else {
    google::protobuf::io::OstreamOutputStream stream(&std::cout);
    google::protobuf::TextFormat::Print(predict_model, &stream);
    google::protobuf::TextFormat::Print(init_model, &stream);
  }
}

void run_bundle() {
  NetDef init_model, predict_model;
  ModelUtil model(init_model, predict_model);

  Keeper(FLAGS_model).AddModel(model, true);
  model.input_dims({3, 4, 5});
  model.output_labels({"A", "B", "C", "D"});

  model.WriteBundle("testbundle.pb");

  NetDef init_model2, predict_model2;
  ModelUtil model2(init_model2, predict_model2);
  model2.ReadBundle("testbundle.pb");

  std::cout << model2.input_dims() << std::endl;
  std::cout << model2.output_labels() << std::endl;
  std::cout << model2.Short();
}

}  // namespace caffe2

int main(int argc, char **argv) {
  caffe2::GlobalInit(&argc, &argv);
  caffe2::run();
  // caffe2::run_bundle();
  google::protobuf::ShutdownProtobufLibrary();
  return 0;
}
