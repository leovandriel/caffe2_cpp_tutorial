#include "caffe2/util/net.h"

#ifdef WITH_CUDA
#include <caffe2/core/context_gpu.h>
#endif

namespace caffe2 {

// Helpers

void NetUtil::AddInput(const std::string input) {
  net.add_external_input(input);
}

void NetUtil::AddOutput(const std::string output) {
  net.add_external_output(output);
}

const std::string& NetUtil::Input(int i) {
  CAFFE_ENFORCE(net.external_input_size() != 0, net.name(),
                " doesn't have any exteral inputs");
  CAFFE_ENFORCE(net.external_input_size() > i, net.name(),
                " is missing exteral input ", i);
  return net.external_input(i);
}
const std::string& NetUtil::Output(int i) {
  CAFFE_ENFORCE(net.external_output_size() != 0, net.name(),
                " doesn't have any exteral outputs");
  CAFFE_ENFORCE(net.external_output_size() > i, net.name(),
                " is missing exteral output ", i);
  return net.external_output(i);
}

void NetUtil::SetName(const std::string name) { net.set_name(name); }

void NetUtil::SetType(const std::string type) { net.set_type(type); }

void NetUtil::SetFillToTrain() {
  for (auto& op : *net.mutable_op()) {
    if (op.type() == "GivenTensorFill") {
      if (op.output(0).find("_w") != std::string::npos) {
        op.set_type("XavierFill");
      }
      if (op.output(0).find("_b") != std::string::npos) {
        op.set_type("ConstantFill");
      }
    }
    op.clear_name();
  }
}

const std::set<std::string> non_inplace_ops({
    "Dropout",  // TODO: see if "they" fixed dropout on cudnn
});

void NetUtil::SetRenameInplace() {
  std::set<std::string> renames;
  for (auto& op : *net.mutable_op()) {
    if (renames.find(op.input(0)) != renames.end()) {
      op.set_input(0, op.input(0) + "_unique");
    }
    if (renames.find(op.output(0)) != renames.end()) {
      renames.erase(op.output(0));
    }
    if (op.input(0) == op.output(0)) {
      if (non_inplace_ops.find(op.type()) != non_inplace_ops.end()) {
        renames.insert(op.output(0));
        op.set_output(0, op.output(0) + "_unique");
      }
    }
  }
}

void NetUtil::SetEngineOps(const std::string engine) {
  for (auto& op : *net.mutable_op()) {
    op.set_engine(engine);
  }
}

void NetUtil::SetDeviceCUDA() {
#ifdef WITH_CUDA
  net.mutable_device_option()->set_device_type(CUDA);
#endif
}

}  // namespace caffe2
