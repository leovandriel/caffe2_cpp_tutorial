#ifndef PRINT_H
#define PRINT_H

#include "caffe2/core/net.h"

namespace caffe2 {

template<typename T> void print(const std::vector<T> vector, const std::string &name = "") {
  if (name.length() > 0) std::cout << name << ": ";
  for (auto &v: vector) {
    std::cout << v << ' ';
  }
  if (name.length() > 0) std::cout << std::endl;
}

void print(const OperatorDef &def) {
  std::cout << "op {" << std::endl;
  for (const auto &input: def.input()) {
    std::cout << "  input: " << '"' << input << '"' << std::endl;
  }
  for (const auto &output: def.output()) {
    std::cout << "  output: " << '"' << output << '"' << std::endl;
  }
  std::cout << "  name: " << '"' << def.name() << '"' << std::endl;
  std::cout << "  type: " << '"' << def.type() << '"' << std::endl;
  if (def.arg_size()) {
    for (const auto &arg: def.arg()) {
      std::cout << "  arg {" << std::endl;
      std::cout << "    name: " << '"' << arg.name() << '"' << std::endl;
      if (arg.has_f()) std::cout << "    f: " << arg.f() << std::endl;
      if (arg.has_i()) std::cout << "    i: " << arg.i() << std::endl;
      if (arg.has_s()) std::cout << "    s: " << '"' << arg.s() << '"' << std::endl;

      if (arg.ints().size() < 10) for (const auto &v: arg.ints()) std::cout << "    ints: " << v << std::endl;
      else std::cout << "    ints: #" << arg.ints().size() << std::endl;
      if (arg.floats().size() < 10) for (const auto &v: arg.floats()) std::cout << "    floats: " << v << std::endl;
      else std::cout << "    floats: #" << arg.floats().size() << std::endl;
      if (arg.strings().size() < 10) for (const auto &v: arg.strings()) std::cout << "    strings: " << '"' << v << '"' << std::endl;
      else std::cout << "    strings: #" << arg.strings().size() << std::endl;
      std::cout << "  }" << std::endl;
    }
  }
  if (def.is_gradient_op()) {
    std::cout << "  is_gradient_op: true" << std::endl;
  }
  std::cout << '}' << std::endl;
}

void print(const NetDef &def) {
  // To just dump the whole thing, use protobuf directly:
  // #include "google/protobuf/io/zero_copy_stream_impl.h"
  // #include "google/protobuf/text_format.h"
  // google::protobuf::io::OstreamOutputStream stream(&std::cout);
  // google::protobuf::TextFormat::Print(init_net, &stream);

  std::cout << "name: " << '"' << def.name() << '"' << std::endl;
  for (const auto &op: def.op()) {
    print(op);
  }
  for (const auto &input: def.external_input()) {
    std::cout << "external_input: " << '"' << input << '"' << std::endl;
  }
  for (const auto &output: def.external_output()) {
    std::cout << "external_output: " << '"' << output << '"' << std::endl;
  }
}

template<typename T, typename C>
void printType(const Tensor<C> &tensor, const std::string &name = "") {
  const auto& data = tensor.template data<T>();
  if (name.length() > 0) std::cout << name << ": ";
  for (auto i = 0; i < (tensor.size() > 100 ? 100 : tensor.size()); ++i) {
    std::cout << data[i] << ' ';
  }
  if (tensor.size() > 100) {
    std::cout << "...";
  }
  if (name.length() > 0) std::cout << std::endl;
}

template<typename C>
void print(const Tensor<C> &tensor, const std::string &name = "") {
  if (tensor.template IsType<float>()) {
    return printType<float>(tensor, name);
  }
  if (tensor.template IsType<int>()) {
    return printType<int>(tensor, name);
  }
  if (tensor.template IsType<long long>()) {
    return printType<long long>(tensor, name);
  }
  std::cout << name << "?" << std::endl;
}

void print(const Blob &blob, const std::string &name = "") {
  print(blob.Get<Tensor<CPUContext>>(), name);
}

template<typename C>
void printBest(const Tensor<C> &tensor, const char **classes, const std::string &name = "") {
    // sort top results
  const auto &probs = tensor.template data<float>();
  std::vector<std::pair<int, int>> pairs;
  for (auto i = 0; i < tensor.size(); i++) {
    if (probs[i] > 0.01) {
      pairs.push_back(std::make_pair(probs[i] * 100, i));
    }
  }
  std:sort(pairs.begin(), pairs.end());

  // show results
  if (name.length() > 0) std::cout << name << ": " << std::endl;
  for (auto pair: pairs) {
    std::cout << pair.first << "% '" << classes[pair.second] << "' (" << pair.second << ")" << std::endl;
  }
}

}  // namespace caffe2

// TensorProto_DataType_UNDEFINED 0
// TensorProto_DataType_FLOAT     1
// TensorProto_DataType_INT32     2
// TensorProto_DataType_BYTE      3
// TensorProto_DataType_STRING    4
// TensorProto_DataType_BOOL      5
// TensorProto_DataType_UINT8     6
// TensorProto_DataType_INT8      7
// TensorProto_DataType_UINT16    8
// TensorProto_DataType_INT16     9
// TensorProto_DataType_INT64    10
// TensorProto_DataType_FLOAT16  12
// TensorProto_DataType_DOUBLE   13

#endif  // PRINT_H
