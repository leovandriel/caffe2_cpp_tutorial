#include "caffe2/core/init.h"
#include "caffe2/core/predictor.h"
#include "caffe2/utils/proto_utils.h"

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "util/models.h"
#include "util/print.h"
#include "util/image.h"
#include "res/imagenet_classes.h"


CAFFE2_DEFINE_string(model, "", "Name of one of the pre-trained models.");
CAFFE2_DEFINE_string(image_file, "res/image_file.jpg", "The image file.");
CAFFE2_DEFINE_int(size_to_fit, 224, "The image file.");
CAFFE2_DEFINE_int(image_mean, 128, "The mean to adjust values to.");

namespace caffe2 {

void run() {
  std::cout << '\n';
  std::cout << "## ImageNet Example ##" << '\n';
  std::cout << '\n';

  if (!FLAGS_model.size()) {
    std::cerr << "specify a model name using --model <name>" << '\n';
    for (auto const &pair: model_lookup) {
      std::cerr << "  " << pair.first << '\n';
    }
    return;
  }

  std::cout << "model_name: " << FLAGS_model << '\n';
  std::cout << "image_file: " << FLAGS_image_file << '\n';
  std::cout << "size_to_fit: " << FLAGS_size_to_fit << '\n';
  std::cout << "image_mean: " << FLAGS_image_mean << '\n';

  std::cout << '\n';

  // load model
  NetDef init_net, predict_net;

  CAFFE_ENFORCE(ensureModel(FLAGS_model), "model ", FLAGS_model, " not found");

  std::string init_filename = "res/" + FLAGS_model + "_init_net.pb";
  std::string predict_filename = "res/" + FLAGS_model + "_predict_net.pb";

  CAFFE_ENFORCE(ReadProtoFromFile(init_filename.c_str(), &init_net));
  CAFFE_ENFORCE(ReadProtoFromFile(predict_filename.c_str(), &predict_net));

  // read image as tensor
  auto input = readImageTensor(FLAGS_image_file, FLAGS_size_to_fit, -FLAGS_image_mean);

  // run net
  Predictor predictor(init_net, predict_net);
  Predictor::TensorVector inputVec({ &input }), outputVec;
  predictor.run(inputVec, &outputVec);

  printBest(*(outputVec[0]), imagenet_classes);
}

}  // namespace caffe2

int main(int argc, char** argv) {
  caffe2::GlobalInit(&argc, &argv);
  caffe2::run();
  google::protobuf::ShutdownProtobufLibrary();
  return 0;
}
