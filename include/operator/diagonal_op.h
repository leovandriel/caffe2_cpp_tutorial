#ifndef DIAGONAL_OP_H
#define DIAGONAL_OP_H

#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename T, class Context>
class DiagonalOp final : public Operator<Context> {
 public:
  DiagonalOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        offset_(OperatorBase::GetRepeatedArgument<TIndex>("offset")) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  bool RunOnDevice() override;

 protected:
  std::vector<TIndex> offset_;
};

template <typename T, class Context>
class DiagonalGradientOp final : public Operator<Context> {
 public:
  DiagonalGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        offset_(OperatorBase::GetRepeatedArgument<TIndex>("offset")) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  bool RunOnDevice() override;

 protected:
  std::vector<TIndex> offset_;
};

} // namespace caffe2

#endif // DIAGONAL_OP_H
