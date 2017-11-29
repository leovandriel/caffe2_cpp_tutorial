#include <caffe2/core/init.h>
#include <caffe2/core/net.h>
#include <caffe2/utils/proto_utils.h>
#include "caffe2/util/blob.h"
#include "caffe2/util/tensor.h"
#include "caffe2/zoo/keeper.h"

#include "caffe2/util/cmd.h"
#include "res/imagenet_classes.h"

CAFFE2_DEFINE_string(model, "", "Name of one of the pre-trained models.");
CAFFE2_DEFINE_string(file, "res/file.jpg", "The image file.");
CAFFE2_DEFINE_string(classes, "", "The class file.")
CAFFE2_DEFINE_int(size, 224, "The image file.");

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

  std::sort(pairs.begin(), pairs.end());

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

  if (!std::ifstream(FLAGS_file).good()) {
    std::cerr << "error: Image file missing: " << FLAGS_file << std::endl;
    return;
  }
  
  // detect classes if specified
  if (FLAGS_classes.size() && !std::ifstream(FLAGS_classes).good()) {
    std::cerr << "error: Class file invalid: " << FLAGS_classes << std::endl;
    return;
  }

  auto cuda = (FLAGS_device != "cpu" && cmd_setup_cuda());

  std::cout << "model: " << FLAGS_model << std::endl;
  std::cout << "file: " << FLAGS_file << std::endl;
  std::cout << "size: " << FLAGS_size << std::endl;
  std::cout << "device: " << FLAGS_device << std::endl;
  std::cout << "using cuda: " << (cuda ? "true" : "false") << std::endl;
  ;

  std::cout << std::endl;

  // read image as tensor
  TensorCPU input;
  TensorUtil(input).ReadImage(FLAGS_file, FLAGS_size);

  std::cout << "loading model.." << std::endl;
  clock_t load_time = 0;
  NetDef init_model, predict_model;
  ModelUtil model(init_model, predict_model);

  // read model files
  load_time -= clock();
  size_t model_size = Keeper(FLAGS_model).AddModel(model, true);
  load_time += clock();

  // set model to use CUDA
  if (FLAGS_device != "cpu") {
    model.SetDeviceCUDA();
  }

  if (FLAGS_dump_model) {
    std::cout << model.Short();
  }

  std::cout << "running model.." << std::endl;
  clock_t predict_time = 0;
  Workspace workspace;

  // setup workspace
  auto &input_name = model.predict.Input(0);
  auto &output_name = model.predict.Output(0);

  CAFFE_ENFORCE(workspace.RunNetOnce(model.init.net));
  CAFFE_ENFORCE(workspace.CreateNet(model.predict.net));

  // run predictor
  BlobUtil(*workspace.GetBlob(input_name)).Set(input);
  predict_time -= clock();
  CAFFE_ENFORCE(workspace.RunNet(model.predict.net.name()));
  predict_time += clock();
  auto output = BlobUtil(*workspace.GetBlob(output_name)).Get();

  std::cout << std::endl;

  // show prediction result using classes
  if(FLAGS_classes.size()){
    std::ifstream file(FLAGS_classes);
    std::string temp;

    std::vector<std::string> strings;

    int i = 0;
    while(std::getline(file, temp)) {
      //Do with temp
      strings.push_back(temp);  
    }

    // convert out vector of strings to char*
    std::vector<const char*> cstrings;
    cstrings.reserve(strings.size());

    for(size_t i = 0; i < strings.size(); ++i)
      cstrings.push_back(const_cast<char*>(strings[i].c_str()));

    printBest(output, &cstrings[0]);
  } else {
    printBest(output, imagenet_classes);    
  }

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
