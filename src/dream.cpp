#include "caffe2/core/init.h"
#include "caffe2/core/net.h"
#include "caffe2/utils/proto_utils.h"

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "util/models.h"
#include "util/print.h"
#include "util/image.h"
#include "util/cuda.h"
#include "util/build.h"
#include "util/net.h"
#include "util/math.h"
#include "res/imagenet_classes.h"
#include "operator/operator_cout.h"


CAFFE2_DEFINE_string(model, "googlenet", "Name of one of the pre-trained models.");
CAFFE2_DEFINE_string(layer, "pool3/3x3_s2", "Name of the layer on which to split the model.");
CAFFE2_DEFINE_int(channel, 3, "The of channel runs.");

CAFFE2_DEFINE_string(image_file, "res/image_file.jpg", "The image file.");
// CAFFE2_DEFINE_string(label, "Chihuahua", "What we're dreaming about.");
CAFFE2_DEFINE_int(train_runs, 200 * caffe2::cuda_multipier, "The of training runs.");
CAFFE2_DEFINE_int(size_to_fit, 224, "The image file.");
CAFFE2_DEFINE_double(learning_rate, 1e-1, "Learning rate.");
CAFFE2_DEFINE_bool(force_cpu, false, "Only use CPU, no CUDA.");

namespace caffe2 {

void AddSuperNaive(NetDef &init_model, NetDef &predict_model, int label_index) {
  // add gradients
  auto &output = predict_model.external_output(0);
  add_xent_ops(predict_model, output);
  add_gradient_ops(predict_model);
  add_iter_lr_ops(init_model, predict_model, FLAGS_learning_rate);

  // add dream operator
  add_constant_fill_int32_op(init_model, { 1 }, label_index, "label");
  predict_model.add_external_input("label");
  add_constant_fill_float_op(init_model, { 1 }, 1.0, "one");
  predict_model.add_external_input("one");
  auto &input = predict_model.external_input(0);
  add_weighted_sum_op(predict_model, { input, "one", input + "_grad", "lr" }, input);
}

void AddNaive(NetDef &init_model, NetDef &predict_model, int channel) {
  auto &output = predict_model.external_output(0);
  add_channel_mean_ops(predict_model, output, 1, 4, channel);
  add_gradient_ops(predict_model);
  add_uniform_fill_float_op(init_model, { 1, 3, FLAGS_size_to_fit, FLAGS_size_to_fit }, -FLAGS_learning_rate / 10, FLAGS_learning_rate / 10, predict_model.external_input(0));
  // add_constant_fill_float_op(init_model, { 1, 3, FLAGS_size_to_fit, FLAGS_size_to_fit }, 1, predict_model.external_input(0));
}

void run() {
  std::cout << std::endl;
  std::cout << "## Deep Dream Example ##" << std::endl;
  std::cout << std::endl;

  if (!FLAGS_model.size()) {
    std::cerr << "specify a model name using --model <name>" << std::endl;
    for (auto const &pair: model_lookup) {
      std::cerr << "  " << pair.first << std::endl;
    }
    return;
  }

  // if (!FLAGS_label.size()) {
  //   std::cerr << "specify a label name using --label <name>" << std::endl;
  //   return;
  // }

  std::cout << "model: " << FLAGS_model << std::endl;
  std::cout << "layer: " << FLAGS_layer << std::endl;
  std::cout << "channel: " << FLAGS_channel << std::endl;

  std::cout << "image_file: " << FLAGS_image_file << std::endl;
  // std::cout << "label: " << FLAGS_label << std::endl;
  std::cout << "train_runs: " << FLAGS_train_runs << std::endl;
  std::cout << "size_to_fit: " << FLAGS_size_to_fit << std::endl;
  std::cout << "learning_rate: " << FLAGS_learning_rate << std::endl;
  std::cout << "force_cpu: " << (FLAGS_force_cpu ? "true" : "false") << std::endl;

  if (!FLAGS_force_cpu) setupCUDA();

  // auto label_index = -1;
  // for (int i = 0; i < 1000; i++) {
  //   if (!strcmp(FLAGS_label.c_str(), imagenet_classes[i])) {
  //     label_index = i;
  //   }
  // }
  // if (label_index < 0) {
  //   for (int i = 0; i < 1000; i++) {
  //     std::cout << "  " << imagenet_classes[i] << std::endl;
  //   }
  //   LOG(FATAL) << "~ image class label not found: " << FLAGS_label;
  // }

  std::cout << std::endl;

  std::cout << "loading model.." << std::endl;
  clock_t load_time = 0;
  NetDef full_init_model, full_predict_model;

  // check if model present
  CAFFE_ENFORCE(ensureModel(FLAGS_model), "model ", FLAGS_model, " not found");
  std::string init_filename = "res/" + FLAGS_model + "_init_net.pb";
  std::string predict_filename = "res/" + FLAGS_model + "_predict_net.pb";

  // read model files
  load_time -= clock();
  CAFFE_ENFORCE(ReadProtoFromFile(init_filename.c_str(), &full_init_model));
  CAFFE_ENFORCE(ReadProtoFromFile(predict_filename.c_str(), &full_predict_model));
  load_time += clock();

  CheckLayerAvailable(full_predict_model, FLAGS_layer);

  NetDef first_init_model, first_predict_model, second_init_model, second_predict_model;
  SplitModel(full_init_model, full_predict_model, FLAGS_layer, first_init_model, first_predict_model, second_init_model, second_predict_model, FLAGS_force_cpu);

  // AddSuperNaive(full_init_model, full_predict_model, label_index);
  AddNaive(first_init_model, first_predict_model, FLAGS_channel);

  // std::cout << "first_init_model -------------" << std::endl;
  // print(first_init_model);
  // std::cout << "first_predict_model -------------" << std::endl;
  // print(first_predict_model);

  // set model to use CUDA
  if (!FLAGS_force_cpu) {
    set_device_cuda_model(first_init_model);
    set_device_cuda_model(first_predict_model);
  }

  std::cout << "running model.." << std::endl;
  clock_t dream_time = 0;
  Workspace workspace;

  // setup workspace
  auto init_net = CreateNet(first_init_model, &workspace);
  auto predict_net = CreateNet(first_predict_model, &workspace);
  init_net->Run();

  auto &input_name = first_predict_model.external_input(0);

  // read image as tensor
  // auto input = readImageTensor(FLAGS_image_file, FLAGS_size_to_fit);
  // set_tensor_blob(*workspace.GetBlob(input_name), input);
  // showImageTensor(input, 0);

  TensorCPU show;
  show.CopyFrom(get_tensor_blob(*workspace.GetBlob("data")));
  normalize_tensor(show, 50);
  showImageTensor(show, 0);

  // run predictor
  for (int i = 1; i <= FLAGS_train_runs; i++) {
    // print(*workspace.GetBlob("data"), "data");
    dream_time -= clock();
    predict_net->Run();
    dream_time += clock();

    auto grad = get_tensor_blob(*workspace.GetBlob("data_grad"));
    float mean = 0, stdev = 0;
    mean_stdev_tensor(grad, &mean, &stdev);
    affine_transform_tensor(grad, FLAGS_learning_rate / (stdev + 1e-8));
    auto data = get_tensor_blob(*workspace.GetBlob("data"));
    add_tensor(data, grad);
    set_tensor_blob(*workspace.GetBlob("data"), data);

    // print(*workspace.GetBlob(FLAGS_layer), FLAGS_layer);
    // print(*workspace.GetBlob("pick"), "pick");
    // print(*workspace.GetBlob("reshape"), "reshape");
    // print(*workspace.GetBlob("score"), "score");
    // print(*workspace.GetBlob("score_grad"), "score_grad");
    // print(*workspace.GetBlob("reshape_grad"), "reshape_grad");
    // print(*workspace.GetBlob("pick_grad"), "pick_grad");
    // print(*workspace.GetBlob(FLAGS_layer + "_grad"), FLAGS_layer + "_grad");
    // print(*workspace.GetBlob("data_grad"), "data_grad");
    // break;


    if (i % (10 * cuda_multipier) == 0) {
      auto score = get_tensor_blob(*workspace.GetBlob("score")).data<float>()[0];
      std::cout << "step: " << i << "  score: " << score << "  mean: " << mean << "  stdev: " << stdev << std::endl;
      if (score == 0) {
        std::cout << "score is zero, try lower learning rate: --learning_rate 1e" << ((int)log10(FLAGS_learning_rate) - 1) << std::endl;
        break;
      }

      TensorCPU show; show.CopyFrom(data);
      normalize_tensor(show, 50);
      showImageTensor(show, 0);
    }
  }

  std::cout << std::endl;

  std::cout << std::setprecision(3)
    << "load: " << ((float)load_time / CLOCKS_PER_SEC)
    << "s  dream: " << ((float)dream_time / CLOCKS_PER_SEC) << "s"
    << std::endl;
}

}  // namespace caffe2

int main(int argc, char** argv) {
  caffe2::GlobalInit(&argc, &argv);
  caffe2::run();
  google::protobuf::ShutdownProtobufLibrary();
  return 0;
}
