#include "caffe2/core/init.h"
#include "caffe2/core/net.h"
#include "caffe2/util/blob.h"
#include "caffe2/util/tensor.h"
#include "caffe2/utils/proto_utils.h"
#include "caffe2/zoo/keeper.h"

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "caffe2/util/misc.h"
#include "res/imagenet_classes.h"

CAFFE2_DEFINE_string(model, "", "Name of one of the pre-trained models.");
CAFFE2_DEFINE_string(layer, "",
                     "Name of the layer on which to split the model.");
CAFFE2_DEFINE_int(channel, 0, "The first channel to run.");
CAFFE2_DEFINE_int(batch, 1, "The number of channels to process in parallel.");
CAFFE2_DEFINE_int(size, 400, "The goal image size.");

CAFFE2_DEFINE_int(train_runs, 60, "The of training runs.");
CAFFE2_DEFINE_int(scale_runs, 10, "The amount of iterations per scale.");
CAFFE2_DEFINE_int(percent_incr, 40, "Percent increase per round.");
CAFFE2_DEFINE_int(initial, -17, "The of initial value.");
CAFFE2_DEFINE_double(learning_rate, 1, "Learning rate.");
CAFFE2_DEFINE_bool(show_image, false, "show image while dreaming.");

#include "caffe2/util/cmd.h"

namespace caffe2 {

void AddNaive(NetDef &init_model, NetDef &dream_model, NetDef &display_model,
              int size) {
  auto &input = dream_model.external_input(0);
  auto &output = dream_model.external_output(0);

  NetUtil init(init_model), dream(dream_model), display(display_model);

  // initialize input data
  init.AddUniformFillOp({FLAGS_batch, 3, size, size}, FLAGS_initial,
                        FLAGS_initial + 1, input);

  // add reduce mean as score
  dream.AddBackMeanOp(output, "mean", 2);
  dream.AddDiagonalOp("mean", "diagonal", {0, FLAGS_channel});
  dream.AddAveragedLossOp("diagonal", "score");
  dream.AddConstantFillWithOp(1.f, "score", "score_grad");

  // add back prop
  dream.AddAllGradientOp();

  // scale gradient
  dream.AddMeanStdevOp(input + "_grad", "_", input + "_grad_stdev");
  dream.AddConstantFillWithOp(0.f, input + "_grad_stdev", "zero");
  dream.AddScaleOp(input + "_grad_stdev", input + "_grad_stdev",
                   1 / FLAGS_learning_rate);
  dream.AddAffineScaleOp(input + "_grad", "zero", input + "_grad_stdev",
                         input + "_grad", true);

  // apply gradient to input data
  init.AddConstantFillOp({1}, 1.f, "one");
  dream.AddInput("one");
  dream.AddWeightedSumOp({input, "one", input + "_grad", "one"}, input);

  // scale data to image
  display.AddMeanStdevOp(input, input + "_mean", input + "_stdev");
  display.AddAffineScaleOp(input, input + "_mean", input + "_stdev", "image",
                           true);
  display.AddScaleOp("image", "image", 25.5f);
  display.AddClipOp("image", "image", -128, 128);
}

void run() {
  if (!cmd_init("Deep Dream Example")) {
    return;
  }

  if (!FLAGS_model.size()) {
    std::cerr << "specify a model name using --model <name>" << std::endl;
    for (auto const &pair : keeper_model_lookup) {
      std::cerr << "  " << pair.first << std::endl;
    }
    return;
  }

  if (!FLAGS_layer.size()) {
    std::cerr << "specify a layer name using --layer <name>" << std::endl;
    return;
  }

  std::cout << "model: " << FLAGS_model << std::endl;
  std::cout << "layer: " << FLAGS_layer << std::endl;
  std::cout << "channel: " << FLAGS_channel << std::endl;
  std::cout << "batch: " << FLAGS_batch << std::endl;
  std::cout << "size: " << FLAGS_size << std::endl;

  std::cout << "train_runs: " << FLAGS_train_runs << std::endl;
  std::cout << "scale_runs: " << FLAGS_scale_runs << std::endl;
  std::cout << "percent_incr: " << FLAGS_percent_incr << std::endl;
  std::cout << "initial: " << FLAGS_initial << std::endl;
  std::cout << "learning_rate: " << FLAGS_learning_rate << std::endl;
  std::cout << "show_image: " << (FLAGS_show_image ? "true" : "false")
            << std::endl;

  std::cout << std::endl;

  std::cout << "loading model.." << std::endl;
  clock_t load_time = 0;
  NetDef base_init_model, base_predict_model;

  // read model files
  load_time -= clock();
  Keeper(FLAGS_model).AddModel(base_init_model, base_predict_model, true);
  load_time += clock();

  // extract dream model
  NetUtil(base_predict_model).CheckLayerAvailable(FLAGS_layer);
  NetDef init_model, dream_model, display_model, unused_model;
  NetUtil init(init_model), dream(dream_model), display(display_model);
  split_model(base_init_model, base_predict_model, FLAGS_layer, init_model,
              dream_model, unused_model, unused_model, FLAGS_device != "cudnn",
              false);

  // add_cout_op(dream_model, { "_conv2/norm2_scale" })->set_engine("CUDNN");

  // add dream operators
  auto image_size = FLAGS_size;
  for (int i = 1; i < FLAGS_train_runs / FLAGS_scale_runs; i++) {
    image_size = image_size * 100 / (100 + FLAGS_percent_incr);
  }
  CHECK(image_size > 10) << "train_runs too high or percent_incr too high";
  AddNaive(init_model, dream_model, display_model, image_size);

  // set model to use CUDA
  if (FLAGS_device != "cpu") {
    init.SetDeviceCUDA();
    dream.SetDeviceCUDA();
    display.SetDeviceCUDA();
    // dream.SetEngineCudnnOps();
  }

  if (FLAGS_dump_model) {
    std::cout << init.Short();
    std::cout << dream.Short();
    std::cout << display.Short();
  }

  std::cout << "running model.." << std::endl;
  clock_t dream_time = 0;
  Workspace workspace;

  // setup workspace
  auto init_net = CreateNet(init_model, &workspace);
  auto predict_net = CreateNet(dream_model, &workspace);
  auto display_net = CreateNet(display_model, &workspace);
  init_net->Run();

  // read image as tensor
  // auto &input_name = dream_model.external_input(0);
  // TensorCPU input;
  // TensorUtil(input).ReadImage(FLAGS_image_file, FLAGS_size / 10);
  // BlobUtil(*workspace.GetBlob(input_name)).Set(input);
  // TensorUtil(input).ShowImages(0);

  std::cout << "start size: " << image_size << std::endl;

  if (FLAGS_show_image) {
    // show current images
    display_net->Run();
    auto image = BlobUtil(*workspace.GetBlob("image")).Get();
    TensorUtil(image).ShowImages("dream");
  }

  // run predictor
  for (auto step = 0; step < FLAGS_train_runs;) {
    // scale up image tiny bit
    image_size =
        std::min(image_size * (100 + FLAGS_percent_incr) / 100, FLAGS_size);
    auto data = BlobUtil(*workspace.GetBlob("data")).Get();
    auto scaled = TensorUtil(data).ScaleImageTensor(image_size, image_size);
    BlobUtil(*workspace.GetBlob("data")).Set(scaled);

    for (int i = 0; i < FLAGS_scale_runs; i++, step++) {
      dream_time -= clock();
      predict_net->Run();
      dream_time += clock();

      if (!step) {
        auto depth = BlobUtil(*workspace.GetBlob(FLAGS_layer)).Get().dim(1);
        std::cout << "channel depth: " << depth << std::endl;

        // print(BlobUtil(*workspace.GetBlob(FLAGS_layer)).Get(), FLAGS_layer);
        // print(BlobUtil(*workspace.GetBlob("mean")).Get(), "mean");
        // print(BlobUtil(*workspace.GetBlob("diagonal")).Get(), "diagonal");
        // print(BlobUtil(*workspace.GetBlob("score")).Get(), "score");
        // print(BlobUtil(*workspace.GetBlob("score_grad")).Get(),
        // "score_grad");
        // print(BlobUtil(*workspace.GetBlob("diagonal_grad")).Get(),
        // "diagonal_grad");
        // print(BlobUtil(*workspace.GetBlob("mean_grad")).Get(), "mean_grad");
        // print(BlobUtil(*workspace.GetBlob(FLAGS_layer +.Get() "_grad")),
        // FLAGS_layer + "_grad");
        // print(BlobUtil(*workspace.GetBlob("data_grad")).Get(), "data_grad");
        // print(BlobUtil(*workspace.GetBlob("data")).Get(), "data");
      }
    }

    auto score = BlobUtil(*workspace.GetBlob("score")).Get().data<float>()[0];
    std::cout << "step: " << step << "  score: " << score
              << "  size: " << image_size << std::endl;

    // show current images
    if (FLAGS_show_image) {
      display_net->Run();
      auto image = BlobUtil(*workspace.GetBlob("image")).Get();
      TensorUtil(image).ShowImages("dream");
    }
  }

  {
    display_net->Run();
    auto image = BlobUtil(*workspace.GetBlob("image")).Get();
    auto safe_layer = FLAGS_layer;
    std::replace(safe_layer.begin(), safe_layer.end(), '/', '_');
    TensorUtil(image).WriteImages("output/" + safe_layer + "_" +
                                  std::to_string(FLAGS_channel));
  }

  std::cout << std::endl;

  std::cout << std::setprecision(3)
            << "load: " << ((float)load_time / CLOCKS_PER_SEC)
            << "s  dream: " << ((float)dream_time / CLOCKS_PER_SEC) << "s"
            << std::endl;
}

}  // namespace caffe2

int main(int argc, char **argv) {
  caffe2::GlobalInit(&argc, &argv);
  caffe2::run();
  google::protobuf::ShutdownProtobufLibrary();
  return 0;
}
