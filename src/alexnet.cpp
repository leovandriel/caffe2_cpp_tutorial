#include "caffe2/core/init.h"
#include "caffe2/core/predictor.h"
#include "caffe2/utils/proto_utils.h"

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "util/print.h"
#include "util/image.h"
#include "res/imagenet_classes.h"

CAFFE2_DEFINE_string(init_net, "res/alexnet_init_net.pb", "The given path to the init protobuffer.");
CAFFE2_DEFINE_string(predict_net, "res/alexnet_predict_net.pb", "The given path to the predict protobuffer.");
CAFFE2_DEFINE_string(image_file, "res/image_file.jpg", "The image file.");
CAFFE2_DEFINE_int(size_to_fit, 227, "The image file.");
CAFFE2_DEFINE_int(image_mean, 128, "The mean to adjust values to.");

namespace caffe2 {

void run() {
  std::cout << '\n';
  std::cout << "## AlexNet Example ##" << '\n';
  std::cout << '\n';

  std::cout << "init_net: " << FLAGS_init_net << '\n';
  std::cout << "predict_net: " << FLAGS_predict_net << '\n';
  std::cout << "image_file: " << FLAGS_image_file << '\n';
  std::cout << "size_to_fit: " << FLAGS_size_to_fit << '\n';
  std::cout << "image_mean: " << FLAGS_image_mean << '\n';

  std::cout << '\n';

  // read image as tensor
  auto input = readImageTensor(FLAGS_image_file, FLAGS_size_to_fit, -FLAGS_image_mean);

  // load model
  NetDef init_net, predict_net;
  CAFFE_ENFORCE(ReadProtoFromFile(FLAGS_init_net, &init_net));
  CAFFE_ENFORCE(ReadProtoFromFile(FLAGS_predict_net, &predict_net));

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
