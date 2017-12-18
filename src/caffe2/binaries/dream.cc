#include <caffe2/core/init.h>
#include <caffe2/core/net.h>
#include <caffe2/utils/proto_utils.h>
#include "caffe2/util/blob.h"
#include "caffe2/util/plot.h"
#include "caffe2/util/tensor.h"
#include "caffe2/util/window.h"
#include "caffe2/zoo/keeper.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

CAFFE2_DEFINE_string(model, "", "Name of one of the pre-trained models.");
CAFFE2_DEFINE_string(layer, "",
                     "Name of the layer on which to split the model.");
CAFFE2_DEFINE_int(channel, -1, "The first channel to run.");
CAFFE2_DEFINE_int(batch, 1, "The number of channels to process in parallel.");
CAFFE2_DEFINE_int(size, 400, "The goal image size.");

CAFFE2_DEFINE_string(file, "", "The image file.");
CAFFE2_DEFINE_int(iters, 60, "The of training runs.");
CAFFE2_DEFINE_int(scale_runs, 10, "The amount of iterations per scale.");
CAFFE2_DEFINE_int(percent_incr, 40, "Percent increase per round.");
CAFFE2_DEFINE_int(initial, -17, "The of initial value.");
CAFFE2_DEFINE_double(lr, 1, "Learning rate.");
CAFFE2_DEFINE_bool(display, false, "Show image while dreaming.");

#include "caffe2/util/cmd.h"

namespace caffe2 {

void AddNaive(ModelUtil &dream, NetUtil &display, int size, int colors) {
  auto &input = dream.predict.Input(0);
  auto &output = dream.predict.Output(0);

  // initialize input data
  dream.init.AddUniformFillOp({FLAGS_batch, colors, size, size}, FLAGS_initial,
                              FLAGS_initial + 1, input);

  // add squared l2 distance to zero as loss
  if (FLAGS_channel >= 0) {
    dream.predict.AddSquaredL2ChannelOp(output, "loss", FLAGS_channel);
  } else {
    dream.predict.AddSquaredL2Op(output, "loss");
  }
  dream.predict.AddConstantFillWithOp(1.f, "loss", "loss_grad");

  if (FLAGS_display) {
    dream.predict.AddTimePlotOp("loss");
  }

  // add back prop
  dream.predict.AddGradientOps();

  // scale gradient
  dream.predict.AddMeanStdevOp(input + "_grad", "_", input + "_grad_stdev");
  dream.predict.AddConstantFillWithOp(0.f, input + "_grad_stdev", "zero");
  dream.predict.AddScaleOp(input + "_grad_stdev", input + "_grad_stdev",
                           1 / FLAGS_lr);
  dream.predict.AddAffineScaleOp(input + "_grad", "zero", input + "_grad_stdev",
                                 input + "_grad", true);

  // apply gradient to input data
  dream.init.AddConstantFillOp({1}, 1.f, "one");
  dream.predict.AddInput("one");
  dream.predict.AddWeightedSumOp({input, "one", input + "_grad", "one"}, input);

  // scale data to image
  if (FLAGS_file.size()) {
    display.AddCopyOp(input, "image");
  } else {
    display.AddMeanStdevOp(input, input + "_mean", input + "_stdev");
    display.AddAffineScaleOp(input, input + "_mean", input + "_stdev", "image",
                             true);
    display.AddScaleOp("image", "image", 25.f);
    display.AddClipOp("image", "image", -128, 128);
  }
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

  std::cout << "iters: " << FLAGS_iters << std::endl;
  std::cout << "scale-runs: " << FLAGS_scale_runs << std::endl;
  std::cout << "percent-incr: " << FLAGS_percent_incr << std::endl;
  std::cout << "initial: " << FLAGS_initial << std::endl;
  std::cout << "lr: " << FLAGS_lr << std::endl;
  std::cout << "display: " << (FLAGS_display ? "true" : "false") << std::endl;

  std::cout << std::endl;

  if (FLAGS_display) {
    auto size =
        std::min(std::max(400, FLAGS_size), (int)sqrt(800000 / FLAGS_batch));
    superWindow("Deep Dream Example");
    moveWindow("loss", 0, 0);
    resizeWindow("loss", size, size);
    setWindowTitle("loss", "loss");
    int x_offset = 1, y_offset = 0;
    for (int i = 0; i < FLAGS_batch; i++) {
      auto name = ("dream-" + std::to_string(i)).c_str();
      moveWindow(name, x_offset * size, y_offset * size);
      resizeWindow(name, size, size);
      setWindowTitle(
          name,
          (FLAGS_layer + " " +
           (FLAGS_channel >= 0 ? std::to_string(FLAGS_channel + i) : "all"))
              .c_str());
      if (++x_offset > 1000 / size) {
        x_offset = 0;
        y_offset++;
      }
    }
  }

  std::cout << "loading model.." << std::endl;
  clock_t load_time = 0;
  NetDef base_init_model, base_predict_model;
  ModelUtil base(base_init_model, base_predict_model);
  // read model files
  load_time -= clock();
  Keeper(FLAGS_model).AddModel(base, true);
  load_time += clock();

  // extract dream model
  base.predict.CheckLayerAvailable(FLAGS_layer);
  NetDef init_model, dream_model, display_model, unused_model;
  NetUtil display(display_model, "display");
  ModelUtil dream(init_model, dream_model);
  ModelUtil unused(unused_model, unused_model);

  base.Split(FLAGS_layer, dream, unused, FLAGS_device != "cudnn", false);

  // add_cout_op(dream.predict.net, { "_conv2/norm2_scale"
  // })->set_engine("CUDNN");

  // add dream operators
  auto image_size = FLAGS_size;
  for (int i = 1; i < FLAGS_iters / FLAGS_scale_runs; i++) {
    image_size = image_size * 100 / (100 + FLAGS_percent_incr);
  }
  if (image_size < 20) {
    image_size = 20;
  }
  auto colors = base.init.net.op(0).arg(0).ints(1);
  AddNaive(dream, display, image_size, colors);

  // set model to use CUDA
  if (FLAGS_device != "cpu") {
    dream.SetDeviceCUDA();
    display.SetDeviceCUDA();
    // dream.SetEngineCudnnOps();
  }

  if (FLAGS_dump_model) {
    std::cout << dream.Short();
    std::cout << display.Short();
  }

  std::cout << "running model.." << std::endl;
  clock_t dream_time = 0;
  Workspace workspace;

  // setup workspace
  CAFFE_ENFORCE(workspace.RunNetOnce(dream.init.net));
  CAFFE_ENFORCE(workspace.CreateNet(dream.predict.net));
  CAFFE_ENFORCE(workspace.CreateNet(display.net));

  // read image as tensor
  if (FLAGS_file.size()) {
    auto &input_name = dream.predict.Input(0);
    TensorCPU input;
    std::vector<int> x;
    TensorUtil(input).ReadImages({FLAGS_file}, image_size, image_size, x, 128);
    BlobUtil(*workspace.GetBlob(input_name)).Set(input);
  }

  std::cout << "start size: " << image_size << std::endl;

  if (FLAGS_display) {
    // show current images
    CAFFE_ENFORCE(workspace.RunNet(display.net.name()));
    auto image = BlobUtil(*workspace.GetBlob("image")).Get();
    TensorUtil(image).ShowImages("dream");

    auto &figure = PlotUtil::Shared("loss");
    figure.Get("rescale").Type(PlotUtil::Vertical).Color(PlotUtil::Gray());
    figure.Show();
  }

  // run predictor
  for (auto step = 0; step < FLAGS_iters;) {
    // scale up image tiny bit
    image_size =
        std::min(image_size * (100 + FLAGS_percent_incr) / 100, FLAGS_size);
    auto data = BlobUtil(*workspace.GetBlob("data")).Get();
    auto scaled = TensorUtil(data).ScaleImageTensor(image_size, image_size);
    BlobUtil(*workspace.GetBlob("data")).Set(scaled);

    for (int i = 0; i < FLAGS_scale_runs; i++, step++) {
      dream_time -= clock();
      CAFFE_ENFORCE(workspace.RunNet(dream.predict.net.name()));
      dream_time += clock();

      if (!step) {
        auto depth = BlobUtil(*workspace.GetBlob(FLAGS_layer)).Get().dim(1);
        std::cout << "channel depth: " << depth << std::endl;

        // BlobUtil(*workspace.GetBlob(FLAGS_layer)).Print(FLAGS_layer, 6000);
        // BlobUtil(*workspace.GetBlob("mean")).Print("mean", 6000);
        // BlobUtil(*workspace.GetBlob("diagonal")).Print("diagonal", 6000);
        // BlobUtil(*workspace.GetBlob("loss")).Print("loss", 6000);
        // BlobUtil(*workspace.GetBlob("loss_grad")).Print("loss_grad", 6000);
        // BlobUtil(*workspace.GetBlob("diagonal_grad"))
        //     .Print("diagonal_grad", 6000);
        // BlobUtil(*workspace.GetBlob("mean_grad")).Print("mean_grad", 6000);
        // BlobUtil(*workspace.GetBlob(FLAGS_layer + "_grad"))
        //     .Print(FLAGS_layer + "_grad", 6000);
        // BlobUtil(*workspace.GetBlob("data_grad")).Print("data_grad", 6000);
        // BlobUtil(*workspace.GetBlob("data")).Print("data", 6000);
      }

      if (step % 10 == 0) {
        auto loss = BlobUtil(*workspace.GetBlob("loss")).Get().data<float>()[0];
        std::cout << "step: " << step << "  loss: " << loss
                  << "  size: " << image_size << std::endl;

        // show current images
        if (FLAGS_display) {
          CAFFE_ENFORCE(workspace.RunNet(display.net.name()));
          auto image = BlobUtil(*workspace.GetBlob("image")).Get();
          TensorUtil(image).ShowImages("dream");
        }
      }
    }
    if (FLAGS_display) {
      auto &figure = PlotUtil::Shared("loss");
      figure.Get("rescale").AddValue(step);
      figure.Show();
    }
  }

  {
    CAFFE_ENFORCE(workspace.RunNet(display.net.name()));
    auto image = BlobUtil(*workspace.GetBlob("image")).Get();
    auto safe_layer = FLAGS_layer;
    std::replace(safe_layer.begin(), safe_layer.end(), '/', '_');
    auto suffix = (FLAGS_channel < 0 ? "_all" : "");
    TensorUtil(image).WriteImages("tmp/" + safe_layer + suffix, 128, false,
                                  FLAGS_channel);
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
