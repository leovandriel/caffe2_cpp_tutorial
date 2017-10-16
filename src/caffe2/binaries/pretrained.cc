#include <caffe2/core/init.h>
#include <caffe2/core/predictor.h>
#include <caffe2/utils/proto_utils.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "res/imagenet_classes.h"

CAFFE2_DEFINE_string(init_net, "res/squeezenet_init_net.pb",
                     "The given path to the init protobuffer.");
CAFFE2_DEFINE_string(predict_net, "res/squeezenet_predict_net.pb",
                     "The given path to the predict protobuffer.");
CAFFE2_DEFINE_string(file, "res/file.jpg", "The image file.");
CAFFE2_DEFINE_int(size, 227, "The image file.");

namespace caffe2 {

void run() {
  std::cout << std::endl;
  std::cout << "## Caffe2 Loading Pre-Trained Models Tutorial ##" << std::endl;
  std::cout << "https://caffe2.ai/docs/zoo.html" << std::endl;
  std::cout << "https://caffe2.ai/docs/tutorial-loading-pre-trained-models.html"
            << std::endl;
  std::cout << "https://caffe2.ai/docs/tutorial-image-pre-processing.html"
            << std::endl;
  std::cout << std::endl;

  if (!std::ifstream(FLAGS_init_net).good() ||
      !std::ifstream(FLAGS_predict_net).good()) {
    std::cerr << "error: Squeezenet model file missing: "
              << (std::ifstream(FLAGS_init_net).good() ? FLAGS_predict_net
                                                       : FLAGS_init_net)
              << std::endl;
    std::cerr << "Make sure to first run ./script/download_resource.sh"
              << std::endl;
    return;
  }

  if (!std::ifstream(FLAGS_file).good()) {
    std::cerr << "error: Image file missing: " << FLAGS_file << std::endl;
    return;
  }

  std::cout << "init-net: " << FLAGS_init_net << std::endl;
  std::cout << "predict-net: " << FLAGS_predict_net << std::endl;
  std::cout << "file: " << FLAGS_file << std::endl;
  std::cout << "size: " << FLAGS_size << std::endl;

  std::cout << std::endl;

  // >>> img =
  // skimage.img_as_float(skimage.io.imread(IMAGE_LOCATION)).astype(np.float32)
  auto image = cv::imread(FLAGS_file);  // CV_8UC3
  std::cout << "image size: " << image.size() << std::endl;

  // scale image to fit
  cv::Size scale(
      std::max(FLAGS_size * image.cols / image.rows, FLAGS_size),
      std::max(FLAGS_size, FLAGS_size * image.rows / image.cols));
  cv::resize(image, image, scale);
  std::cout << "scaled size: " << image.size() << std::endl;

  // crop image to fit
  cv::Rect crop((image.cols - FLAGS_size) / 2,
                (image.rows - FLAGS_size) / 2, FLAGS_size,
                FLAGS_size);
  image = image(crop);
  std::cout << "cropped size: " << image.size() << std::endl;

  // convert to float, normalize to mean 128
  image.convertTo(image, CV_32FC3, 1.0, -128);
  std::cout << "value range: ("
            << *std::min_element((float *)image.datastart,
                                 (float *)image.dataend)
            << ", "
            << *std::max_element((float *)image.datastart,
                                 (float *)image.dataend)
            << ")" << std::endl;

  // convert NHWC to NCHW
  vector<cv::Mat> channels(3);
  cv::split(image, channels);
  std::vector<float> data;
  for (auto &c : channels) {
    data.insert(data.end(), (float *)c.datastart, (float *)c.dataend);
  }
  std::vector<TIndex> dims({1, image.channels(), image.rows, image.cols});
  TensorCPU input(dims, data, NULL);

  // Load Squeezenet model
  NetDef init_net, predict_net;

  // >>> with open(path_to_INIT_NET) as f:
  CAFFE_ENFORCE(ReadProtoFromFile(FLAGS_init_net, &init_net));

  // >>> with open(path_to_PREDICT_NET) as f:
  CAFFE_ENFORCE(ReadProtoFromFile(FLAGS_predict_net, &predict_net));

  // >>> p = workspace.Predictor(init_net, predict_net)
  Predictor predictor(init_net, predict_net);

  // >>> results = p.run([img])
  Predictor::TensorVector inputVec({&input}), outputVec;
  predictor.run(inputVec, &outputVec);
  auto &output = *(outputVec[0]);

  // sort top results
  const auto &probs = output.data<float>();
  std::vector<std::pair<int, int>> pairs;
  for (auto i = 0; i < output.size(); i++) {
    if (probs[i] > 0.01) {
      pairs.push_back(std::make_pair(probs[i] * 100, i));
    }
  }

  std::sort(pairs.begin(), pairs.end());

  std::cout << std::endl;

  // show results
  std::cout << "output: " << std::endl;
  for (auto pair : pairs) {
    std::cout << "  " << pair.first << "% '" << imagenet_classes[pair.second]
              << "' (" << pair.second << ")" << std::endl;
  }
}

}  // namespace caffe2

int main(int argc, char **argv) {
  caffe2::GlobalInit(&argc, &argv);
  caffe2::run();
  google::protobuf::ShutdownProtobufLibrary();
  return 0;
}
