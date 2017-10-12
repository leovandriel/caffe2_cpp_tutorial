#ifndef CMD_H
#define CMD_H

#ifdef WITH_CUDA
#include <caffe2/core/context_gpu.h>
#endif

CAFFE2_DEFINE_string(optimizer, "adam",
                     "Training optimizer: sgd/momentum/adagrad/adam");
CAFFE2_DEFINE_string(device, "cudnn", "Computation device: cpu/cuda/cudnn");
CAFFE2_DEFINE_bool(dump_model, false, "output dream model.");

static const std::set<std::string> device_types({"cpu", "cuda", "cudnn"});
static const std::set<std::string> optimizer_types({"sgd", "momentum",
                                                    "adagrad", "adam",
                                                    "rmsprop"});

namespace caffe2 {

bool cmd_setup_cuda() {
  DeviceOption option;
  option.set_device_type(CUDA);
#ifdef WITH_CUDA
  new CUDAContext(option);
  return true;
#else
  return false;
#endif
}

bool cmd_init(const std::string title) {
  std::cout << std::endl;
  std::cout << "## " << title << " ##" << std::endl;
  std::cout << std::endl;

  if (device_types.find(FLAGS_device) == device_types.end()) {
    std::cerr << "incorrect device type ("
              << std::vector<std::string>(device_types.begin(),
                                          device_types.end())
              << "): " << FLAGS_device << std::endl;
    return false;
  }

  if (optimizer_types.find(FLAGS_optimizer) == optimizer_types.end()) {
    std::cerr << "incorrect optimizer type ("
              << std::vector<std::string>(optimizer_types.begin(),
                                          optimizer_types.end())
              << "): " << FLAGS_optimizer << std::endl;
    return false;
  }

  auto cuda = (FLAGS_device != "cpu" && cmd_setup_cuda());

  std::cout << "optimizer: " << FLAGS_optimizer << std::endl;
  std::cout << "device: " << FLAGS_device << std::endl;
  std::cout << "using cuda: " << (cuda ? "true" : "false") << std::endl;
  ;
  std::cout << "dump-model: " << (FLAGS_dump_model ? "true" : "false")
            << std::endl;

  return true;
}

}  // namespace caffe2

#endif  // CMD_H
