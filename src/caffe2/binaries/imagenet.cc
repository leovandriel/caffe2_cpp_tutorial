#include "caffe2/core/init.h"
#include "caffe2/core/net.h"
#include "caffe2/utils/proto_utils.h"

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "util/zoo.h"
#include "util/print.h"
#include "util/image.h"
#include "util/cuda.h"
#include "res/imagenet_classes.h"


CAFFE2_DEFINE_string(model, "", "Name of one of the pre-trained models.");
CAFFE2_DEFINE_string(image_file, "res/image_file.jpg", "The image file.");
CAFFE2_DEFINE_int(size_to_fit, 224, "The image file.");
CAFFE2_DEFINE_bool(force_cpu, false, "Only use CPU, no CUDA.");

namespace caffe2 {

void run() {
  std::cout << std::endl;
  std::cout << "## ImageNet Example ##" << std::endl;
  std::cout << std::endl;

  if (!FLAGS_model.size()) {
    std::cerr << "specify a model name using --model <name>" << std::endl;
    for (auto const &pair: model_lookup) {
      std::cerr << "  " << pair.first << std::endl;
    }
    return;
  }

  if (!std::ifstream(FLAGS_image_file).good()) {
    std::cerr << "error: Image file missing: " << FLAGS_image_file << std::endl;
    return;
  }

  std::cout << "model: " << FLAGS_model << std::endl;
  std::cout << "image_file: " << FLAGS_image_file << std::endl;
  std::cout << "size_to_fit: " << FLAGS_size_to_fit << std::endl;
  std::cout << "force_cpu: " << (FLAGS_force_cpu ? "true" : "false") << std::endl;

  if (!FLAGS_force_cpu) setupCUDA();

  std::cout << std::endl;

  // read image as tensor
  auto input = readImageTensor(FLAGS_image_file, FLAGS_size_to_fit);

  std::cout << "loading model.." << std::endl;
  clock_t load_time = 0;
  NetDef init_model, predict_model;

  // check if model present
  CAFFE_ENFORCE(ensureModel(FLAGS_model), "model ", FLAGS_model, " not found");
  std::string init_filename = "res/" + FLAGS_model + "_init_net.pb";
  std::string predict_filename = "res/" + FLAGS_model + "_predict_net.pb";

  // get model size
  auto init_size = std::ifstream(init_filename, std::ifstream::ate | std::ifstream::binary).tellg();
  auto predict_size = std::ifstream(predict_filename, std::ifstream::ate | std::ifstream::binary).tellg();
  auto model_size = init_size + predict_size;

  // read model files
  load_time -= clock();
  CAFFE_ENFORCE(ReadProtoFromFile(init_filename.c_str(), &init_model));
  CAFFE_ENFORCE(ReadProtoFromFile(predict_filename.c_str(), &predict_model));
  load_time += clock();

  // set model to use CUDA
  if (!FLAGS_force_cpu) {
    set_device_cuda_model(init_model);
    set_device_cuda_model(predict_model);
  }

  std::cout << "running model.." << std::endl;
  clock_t predict_time = 0;
  Workspace workspace;

  // setup workspace
  auto &input_name = predict_model.external_input(0);
  auto &output_name = predict_model.external_output(0);
  auto init_net = CreateNet(init_model, &workspace);
  auto predict_net = CreateNet(predict_model, &workspace);
  init_net->Run();

  // run predictor
  set_tensor_blob(*workspace.GetBlob(input_name), input);
  predict_time -= clock();
  predict_net->Run();
  predict_time += clock();
  auto output = get_tensor_blob(*workspace.GetBlob(output_name));

  std::cout << std::endl;

  // show prediction result
  printBest(output, imagenet_classes);

  std::cout << std::endl;

  std::cout << std::setprecision(3)
    << "load: " << ((float)load_time / CLOCKS_PER_SEC)
    << "s  predict: " << ((float)predict_time / CLOCKS_PER_SEC)
    << "s  model: " << ((float)model_size / 1000000) << "MB"
    << std::endl;
}

}  // namespace caffe2

int main(int argc, char** argv) {
  caffe2::GlobalInit(&argc, &argv);
  caffe2::run();
  google::protobuf::ShutdownProtobufLibrary();
  return 0;
}
