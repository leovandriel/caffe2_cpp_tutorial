#include "caffe2/core/init.h"
#include "caffe2/core/net.h"
#include "caffe2/util/blob.h"
#include "caffe2/util/tensor.h"
#include "caffe2/utils/proto_utils.h"
#include "caffe2/zoo/keeper.h"

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "caffe2/util/cmd.h"
#include "res/imagenet_classes.h"

CAFFE2_DEFINE_string(model, "", "Name of one of the pre-trained models.");
CAFFE2_DEFINE_string(image_file, "res/image_file.jpg", "The image file.");
CAFFE2_DEFINE_int(size_to_fit, 224, "The image file.");

namespace caffe2 {

template <typename C>
void printBest(const Tensor<C> &tensor, const char **classes,
               const std::string &name = "") {
  // sort top results
  const auto &probs = tensor.template data<float>();
  std::vector<std::pair<int, int>> pairs;
  for (auto i = 0; i < tensor.size(); i++) {
    if (probs[i] > 0.01) {
      pairs.push_back(std::make_pair(probs[i] * 100, i));
    }
  }
std:
  sort(pairs.begin(), pairs.end());

  // show results
  if (name.length() > 0) std::cout << name << ": " << std::endl;
  for (auto pair : pairs) {
    std::cout << pair.first << "% '" << classes[pair.second] << "' ("
              << pair.second << ")" << std::endl;
  }
}

void run() {
  std::cout << std::endl;
  std::cout << "## ImageNet Example ##" << std::endl;
  std::cout << std::endl;

  if (!FLAGS_model.size()) {
    std::cerr << "specify a model name using --model <name>" << std::endl;
    for (auto const &pair : keeper_model_lookup) {
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
  std::cout << "device: " << FLAGS_device << std::endl;

  if (FLAGS_device != "cpu") cmd_setup_cuda();

  std::cout << std::endl;

  // read image as tensor
  TensorCPU input;
  TensorUtil(input).ReadImage(FLAGS_image_file, FLAGS_size_to_fit);

  std::cout << "loading model.." << std::endl;
  clock_t load_time = 0;
  NetDef init_model, predict_model;
  NetUtil init(init_model), predict(predict_model);

  // read model files
  load_time -= clock();
  Keeper(FLAGS_model).AddModel(init_model, predict_model, true);
  load_time += clock();

  // get model size
  auto init_size = std::ifstream("res/" + FLAGS_model + "_init_net.pb",
                                 std::ifstream::ate | std::ifstream::binary)
                       .tellg();
  auto predict_size = std::ifstream("res/" + FLAGS_model + "_predict_net.pb",
                                    std::ifstream::ate | std::ifstream::binary)
                          .tellg();
  auto model_size = init_size + predict_size;

  // set model to use CUDA
  if (FLAGS_device != "cpu") {
    init.SetDeviceCUDA();
    predict.SetDeviceCUDA();
  }

  if (FLAGS_dump_model) {
    std::cout << init.Short();
    std::cout << predict.Short();
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
  BlobUtil(*workspace.GetBlob(input_name)).Set(input);
  predict_time -= clock();
  predict_net->Run();
  predict_time += clock();
  auto output = BlobUtil(*workspace.GetBlob(output_name)).Get();

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

int main(int argc, char **argv) {
  caffe2::GlobalInit(&argc, &argv);
  caffe2::run();
  google::protobuf::ShutdownProtobufLibrary();
  return 0;
}
