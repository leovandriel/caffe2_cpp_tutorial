#ifndef OPERATOR_COUT_OP_H
#define OPERATOR_COUT_OP_H

#include <caffe2/core/operator.h>

namespace caffe2 {

template <class Context>
class CoutOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  CoutOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws), titles(def.input_size()) {
    for (auto i = 0; i < titles.size(); i++) {
      titles[i] = def.input(i);
    }
  }

  bool RunOnDevice() override;

 protected:
  std::vector<std::string> titles;
};

}  // namespace caffe2

#endif  // OPERATOR_COUT_OP_H
