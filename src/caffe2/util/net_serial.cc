#include "caffe2/util/net.h"

#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

namespace caffe2 {

std::string NetUtil::Proto() {
  std::string s;
  google::protobuf::io::StringOutputStream stream(&s);
  google::protobuf::TextFormat::Print(net, &stream);
  return s;
}

template <typename T>
std::string net_short_values(const T& values,
                             const std::string& separator = " ",
                             const std::string& prefix = "[",
                             const std::string& suffix = "]",
                             int collapse = 64) {
  std::stringstream stream;
  if (values.size() > collapse) {
    stream << "[#" << values.size() << "]";
  } else {
    bool is_next = false;
    for (const auto& v : values) {
      if (is_next) {
        stream << separator;
      } else {
        stream << prefix;
        is_next = true;
      }
      stream << v;
    }
    if (is_next) {
      stream << suffix;
    }
  }
  return stream.str();
}

std::string net_short_op(const OperatorDef& def) {
  std::stringstream stream;
  stream << "op: " << def.type();
  if (def.name().size()) {
    stream << " " << '"' << def.name() << '"';
  }
  if (def.input_size() || def.output_size()) {
    stream << " " << net_short_values(def.input(), " ", "(", ")") << "->"
           << net_short_values(def.output(), " ", "(", ")");
  }
  if (def.arg_size()) {
    for (const auto& arg : def.arg()) {
      stream << " " << arg.name() << ":";
      if (arg.has_f()) {
        stream << arg.f();
      } else if (arg.has_i()) {
        stream << arg.i();
      } else if (arg.has_s()) {
        stream << arg.s();
      } else {
        stream << net_short_values(arg.ints());
        stream << net_short_values(arg.floats());
        stream << net_short_values(arg.strings());
      }
    }
  }
  if (def.has_engine()) {
    stream << " engine:" << def.engine();
  }
  if (def.has_device_option() && def.device_option().has_device_type()) {
    stream << " device_type:" << def.device_option().device_type();
  }
  if (def.has_device_option() && def.device_option().has_cuda_gpu_id()) {
    stream << " cuda_gpu_id:" << def.device_option().cuda_gpu_id();
  }
  if (def.has_is_gradient_op()) {
    stream << " is_gradient_op:true";
  }
  stream << std::endl;
  return stream.str();
}

std::string net_short_net(const NetDef& def) {
  std::stringstream stream;
  stream << "net: ------------- " << def.name() << " -------------"
         << std::endl;
  for (const auto& op : def.op()) {
    stream << net_short_op(op);
  }
  if (def.has_device_option() && def.device_option().has_device_type()) {
    stream << "device_type:" << def.device_option().device_type() << std::endl;
  }
  if (def.has_device_option() && def.device_option().has_cuda_gpu_id()) {
    stream << "cuda_gpu_id:" << def.device_option().cuda_gpu_id() << std::endl;
  }
  if (def.has_num_workers()) {
    stream << "num_workers: " << def.num_workers() << std::endl;
  }
  if (def.external_input_size()) {
    stream << "external_input: " << net_short_values(def.external_input())
           << std::endl;
  }
  if (def.external_output_size()) {
    stream << "external_output: " << net_short_values(def.external_output())
           << std::endl;
  }
  return stream.str();
}

std::string NetUtil::Short() { return net_short_net(net); }

void NetUtil::Print() {
  google::protobuf::io::OstreamOutputStream stream(&std::cout);
  google::protobuf::TextFormat::Print(net, &stream);
}

size_t NetUtil::Write(const std::string& path) const {
  WriteProtoToBinaryFile(net, path);
  return std::ifstream(path, std::ifstream::ate | std::ifstream::binary)
      .tellg();
}

size_t NetUtil::WriteText(const std::string& path) const {
  WriteProtoToTextFile(net, path);
  return std::ifstream(path, std::ifstream::ate | std::ifstream::binary)
      .tellg();
}

size_t NetUtil::Read(const std::string& path) {
  CAFFE_ENFORCE(ReadProtoFromFile(path.c_str(), &net));
  return std::ifstream(path, std::ifstream::ate | std::ifstream::binary)
      .tellg();
}

#include <fcntl.h>
#include <cerrno>
#include <fstream>

// dot -Tsvg -omodel.svg model.gv
size_t NetUtil::WriteGraph(const std::string& path) const {
  std::ofstream file(path);
  if (file.is_open()) {
    file << "digraph {" << std::endl;
    auto index = 0;
    file << '\t' << "node [shape=box];";
    for (const auto& op : net.op()) {
      auto name = op.type() + '_' + std::to_string(index++);
      file << ' ' << name;
      // if (index > 250) break;
    }
    file << ';';
    file << '\t' << "node [shape=oval];";
    index = 0;
    for (const auto& op : net.op()) {
      auto name = op.type() + '_' + std::to_string(index++);
      for (const auto& input : op.input()) {
        file << '\t' << '"' << input << '"' << " -> " << '"' << name << '"'
             << ';' << std::endl;
      }
      for (const auto& output : op.output()) {
        file << '\t' << '"' << name << '"' << " -> " << '"' << output << '"'
             << ';' << std::endl;
      }
      // if (index > 250) break;
    }
    // file << "overlap=false" << std::endl;
    file << "}" << std::endl;
    file.close();
  }
  return std::ifstream(path, std::ifstream::ate | std::ifstream::binary)
      .tellg();
}

}  // namespace caffe2
