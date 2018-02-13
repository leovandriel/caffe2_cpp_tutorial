#include "caffe2/util/net.h"

namespace caffe2 {

OperatorDef* NetUtil::AddOp(const std::string& name,
                            const std::vector<std::string>& inputs,
                            const std::vector<std::string>& outputs) {
  auto op = net.add_op();
  op->set_type(name);
  for (auto input : inputs) {
    op->add_input(input);
  }
  for (auto output : outputs) {
    op->add_output(output);
  }
  return op;
}

Argument* net_add_arg(OperatorDef& op, const std::string& name) {
  auto arg = op.add_arg();
  arg->set_name(name);
  return arg;
}

Argument* net_add_arg(OperatorDef& op, const std::string& name, int value) {
  auto arg = net_add_arg(op, name);
  arg->set_i(value);
  return arg;
}

Argument* net_add_arg(OperatorDef& op, const std::string& name, float value) {
  auto arg = net_add_arg(op, name);
  arg->set_f(value);
  return arg;
}

Argument* net_add_arg(OperatorDef& op, const std::string& name,
                      std::vector<int> values) {
  auto arg = net_add_arg(op, name);
  for (auto value : values) {
    arg->add_ints(value);
  }
  return arg;
}

Argument* net_add_arg(OperatorDef& op, const std::string& name,
                      std::vector<TIndex> values) {
  auto arg = net_add_arg(op, name);
  for (auto value : values) {
    arg->add_ints(value);
  }
  return arg;
}

Argument* net_add_arg(OperatorDef& op, const std::string& name,
                      std::vector<float> values) {
  auto arg = net_add_arg(op, name);
  for (auto value : values) {
    arg->add_floats(value);
  }
  return arg;
}

Argument* net_add_arg(OperatorDef& op, const std::string& name,
                      const std::string& value) {
  auto arg = net_add_arg(op, name);
  arg->set_s(value);
  return arg;
}

Argument* net_add_arg(OperatorDef& op, const std::string& name,
                      const std::vector<std::string>& values) {
  auto arg = net_add_arg(op, name);
  for (auto value : values) {
    arg->add_strings(value);
  }
  return arg;
}

// I/O

OperatorDef* NetUtil::AddCreateDbOp(const std::string& reader,
                                    const std::string& db_type,
                                    const std::string& db_path) {
  auto op = AddOp("CreateDB", {}, {reader});
  net_add_arg(*op, "db_type", db_type);
  net_add_arg(*op, "db", db_path);
  return op;
}

OperatorDef* NetUtil::AddTensorProtosDbInputOp(const std::string& reader,
                                               const std::string& data,
                                               const std::string& label,
                                               int batch_size) {
  auto op = AddOp("TensorProtosDBInput", {reader}, {data, label});
  net_add_arg(*op, "batch_size", batch_size);
  return op;
}

OperatorDef* NetUtil::AddCoutOp(const std::vector<std::string>& params) {
  return AddOp("Cout", params, {});
}

OperatorDef* NetUtil::AddZeroOneOp(const std::string& pred,
                                   const std::string& label) {
  return AddOp("ZeroOne", {pred, label}, {});
}

OperatorDef* NetUtil::AddShowWorstOp(const std::string& pred,
                                     const std::string& label,
                                     const std::string& data, float scale,
                                     float mean) {
  auto op = AddOp("ShowWorst", {pred, label, data}, {});
  net_add_arg(*op, "scale", scale);
  net_add_arg(*op, "mean", mean);
  return op;
}

OperatorDef* NetUtil::AddTimePlotOp(const std::string& data,
                                    const std::string& iter,
                                    const std::string& window,
                                    const std::string& label, int step) {
  std::vector<std::string> inputs = {data};
  if (iter.size()) {
    inputs.push_back(iter);
  }
  auto op = AddOp("TimePlot", inputs, {});
  if (window.size()) {
    net_add_arg(*op, "window", window);
  }
  if (label.size()) {
    net_add_arg(*op, "label", label);
  }
  if (step) {
    net_add_arg(*op, "step", step);
  }
  return op;
}

OperatorDef* NetUtil::AddEnsureCpuOutputOp(const std::string& input,
                                           const std::string& output) {
  return AddOp("EnsureCPUOutput", {input}, {output});
}

OperatorDef* NetUtil::AddCopyFromCpuInputOp(const std::string& input,
                                            const std::string& output) {
  return AddOp("CopyFromCPUInput", {input}, {output});
}

OperatorDef* NetUtil::AddCopyOp(const std::string& input,
                                const std::string& output) {
  return AddOp("Copy", {input}, {output});
}

OperatorDef* NetUtil::AddCreateMutexOp(const std::string& param) {
  return AddOp("CreateMutex", {}, {param});
}

OperatorDef* NetUtil::AddPrintOp(const std::string& param, bool to_file) {
  auto op = AddOp("Print", {param}, {});
  if (to_file) {
    net_add_arg(*op, "to_file", 1);
  }
  return op;
}

OperatorDef* NetUtil::AddSummarizeOp(const std::string& param, bool to_file) {
  auto op = AddOp("Summarize", {param}, {});
  if (to_file) {
    net_add_arg(*op, "to_file", 1);
  }
  return op;
}

// Initialization

OperatorDef* NetUtil::AddConstantFillOp(const std::vector<int>& shape,
                                        const std::string& param) {
  auto op = AddOp("ConstantFill", {}, {param});
  net_add_arg(*op, "shape", shape);
  return op;
}

OperatorDef* NetUtil::AddXavierFillOp(const std::vector<int>& shape,
                                      const std::string& param) {
  auto op = AddOp("XavierFill", {}, {param});
  net_add_arg(*op, "shape", shape);
  return op;
}

OperatorDef* NetUtil::AddMSRAFillOp(const std::vector<int>& shape,
                                    const std::string& param) {
  auto op = AddOp("MSRAFill", {}, {param});
  net_add_arg(*op, "shape", shape);
  return op;
}

OperatorDef* NetUtil::AddUniformFillOp(const std::vector<int>& shape, float min,
                                       float max, const std::string& param) {
  auto op = AddOp("UniformFill", {}, {param});
  net_add_arg(*op, "shape", shape);
  net_add_arg(*op, "min", min);
  net_add_arg(*op, "max", max);
  return op;
}

OperatorDef* NetUtil::AddGausianFillOp(const std::vector<int>& shape,
                                       float mean, float std,
                                       const std::string& param) {
  auto op = AddOp("GaussianFill", {}, {param});
  net_add_arg(*op, "shape", shape);
  net_add_arg(*op, "mean", mean);
  net_add_arg(*op, "std", std);
  return op;
}

OperatorDef* NetUtil::AddConstantFillOp(const std::vector<int>& shape,
                                        float value, const std::string& param) {
  auto op = AddOp("ConstantFill", {}, {param});
  net_add_arg(*op, "shape", shape);
  net_add_arg(*op, "value", value);
  return op;
}

OperatorDef* NetUtil::AddConstantFillOp(const std::vector<int>& shape,
                                        int64_t value,
                                        const std::string& param) {
  auto op = AddOp("ConstantFill", {}, {param});
  net_add_arg(*op, "shape", shape);
  net_add_arg(*op, "value", (int)value);
  net_add_arg(*op, "dtype", TensorProto_DataType_INT64);
  return op;
}

OperatorDef* NetUtil::AddConstantFillWithOp(float value,
                                            const std::string& input,
                                            const std::string& output) {
  auto op = AddOp("ConstantFill", {input}, {output});
  net_add_arg(*op, "value", value);
  return op;
}

OperatorDef* NetUtil::AddVectorFillOp(const std::vector<int>& values,
                                      const std::string& name) {
  auto op = AddOp("GivenTensorFill", {}, {name});
  net_add_arg(*op, "shape", std::vector<int>({(int)values.size()}));
  net_add_arg(*op, "values", values);
  net_add_arg(*op, "dtype", TensorProto_DataType_INT32);
  return op;
}

OperatorDef* NetUtil::AddGivenTensorFillOp(const TensorCPU& tensor,
                                           const std::string& name) {
  auto op = AddOp("GivenTensorFill", {}, {name});
  net_add_arg(*op, "shape", tensor.dims());
  auto arg = net_add_arg(*op, "values");
  const auto& data = tensor.data<float>();
  for (auto i = 0; i < tensor.size(); ++i) {
    arg->add_floats(data[i]);
  }
  return op;
}

// Prediction

OperatorDef* NetUtil::AddConvOp(const std::string& input, const std::string& w,
                                const std::string& b, const std::string& output,
                                int stride, int padding, int kernel, int group,
                                const std::string& order) {
  auto op = AddOp("Conv",
                  b.size() ? std::vector<std::string>({input, w, b})
                           : std::vector<std::string>({input, w}),
                  {output});
  net_add_arg(*op, "stride", stride);
  net_add_arg(*op, "pad", padding);
  net_add_arg(*op, "kernel", kernel);
  if (group != 0) net_add_arg(*op, "group", group);
  net_add_arg(*op, "order", order);
  return op;
}

OperatorDef* NetUtil::AddReluOp(const std::string& input,
                                const std::string& output) {
  return AddOp("Relu", {input}, {output});
}

OperatorDef* NetUtil::AddLeakyReluOp(const std::string& input,
                                     const std::string& output, float alpha) {
  auto op = AddOp("LeakyRelu", {input}, {output});
  net_add_arg(*op, "alpha", alpha);
  return op;
}

OperatorDef* NetUtil::AddSigmoidOp(const std::string& input,
                                   const std::string& output) {
  auto op = AddOp("Sigmoid", {input}, {output});
  return op;
}

OperatorDef* NetUtil::AddLrnOp(const std::string& input,
                               const std::string& output, int size, float alpha,
                               float beta, float bias,
                               const std::string& order) {
  auto op = AddOp("LRN", {input}, {output, "_" + output + "_scale"});
  net_add_arg(*op, "size", size);
  net_add_arg(*op, "alpha", alpha);
  net_add_arg(*op, "beta", beta);
  net_add_arg(*op, "bias", bias);
  net_add_arg(*op, "order", order);
  return op;
}

OperatorDef* NetUtil::AddMaxPoolOp(const std::string& input,
                                   const std::string& output, int stride,
                                   int padding, int kernel,
                                   const std::string& order) {
  auto op = AddOp("MaxPool", {input}, {output});
  net_add_arg(*op, "stride", stride);
  net_add_arg(*op, "pad", padding);
  net_add_arg(*op, "kernel", kernel);
  net_add_arg(*op, "order", order);
  net_add_arg(*op, "legacy_pad", 3);
  return op;
}

OperatorDef* NetUtil::AddAveragePoolOp(const std::string& input,
                                       const std::string& output, int stride,
                                       int padding, int kernel,
                                       const std::string& order) {
  auto op = AddOp("AveragePool", {input}, {output});
  net_add_arg(*op, "stride", stride);
  net_add_arg(*op, "pad", padding);
  net_add_arg(*op, "kernel", kernel);
  net_add_arg(*op, "order", order);
  net_add_arg(*op, "legacy_pad", 3);
  return op;
}

OperatorDef* NetUtil::AddFcOp(const std::string& input, const std::string& w,
                              const std::string& b, const std::string& output,
                              int axis) {
  auto op = AddOp("FC", {input, w, b}, {output});
  if (axis != 1) {
    net_add_arg(*op, "axis", axis);
  }
  return op;
}

OperatorDef* NetUtil::AddDropoutOp(const std::string& input,
                                   const std::string& output, float ratio) {
  auto op = AddOp("Dropout", {input}, {output, "_" + output + "_mask"});
  net_add_arg(*op, "ratio", ratio);
  net_add_arg(*op, "is_test", 1);  // TODO
  return op;
}

OperatorDef* NetUtil::AddSoftmaxOp(const std::string& input,
                                   const std::string& output, int axis) {
  auto op = AddOp("Softmax", {input}, {output});
  if (axis != 1) {
    net_add_arg(*op, "axis", axis);
  }
  return op;
}

OperatorDef* NetUtil::AddConcatOp(const std::vector<std::string>& inputs,
                                  const std::string& output,
                                  const std::string& order) {
  auto op = AddOp("Concat", inputs, {output, "_" + output + "_dims"});
  net_add_arg(*op, "order", order);
  return op;
}

OperatorDef* NetUtil::AddSpatialBNOp(const std::vector<std::string>& inputs,
                                     const std::vector<std::string>& outputs,
                                     float epsilon, float momentum, bool test,
                                     const std::string& order) {
  auto op = AddOp("SpatialBN", inputs, outputs);
  net_add_arg(*op, "is_test", test);  // TODO
  net_add_arg(*op, "epsilon", epsilon);
  net_add_arg(*op, "momentum", momentum);
  net_add_arg(*op, "order", order);
  return op;
}

OperatorDef* NetUtil::AddSumOp(const std::vector<std::string>& inputs,
                               const std::string& sum) {
  return AddOp("Sum", inputs, {sum});
}

OperatorDef* NetUtil::AddMulOp(const std::vector<std::string>& inputs,
                               const std::string& output, int axis,
                               int broadcast) {
  auto op = AddOp("Mul", inputs, {output});
  net_add_arg(*op, "axis", axis);
  net_add_arg(*op, "broadcast", broadcast);
  return op;
}

OperatorDef* NetUtil::AddAddOp(const std::vector<std::string>& inputs,
                               const std::string& output, int axis,
                               int broadcast) {
  auto op = AddOp("Add", inputs, {output});
  net_add_arg(*op, "axis", axis);
  net_add_arg(*op, "broadcast", broadcast);
  return op;
}

OperatorDef* NetUtil::AddLSTMUnitOp(const std::vector<std::string>& inputs,
                                    const std::vector<std::string>& outputs,
                                    int drop_states, float forget_bias) {
  auto op = AddOp("LSTMUnit", inputs, outputs);
  net_add_arg(*op, "drop_states", drop_states);
  net_add_arg(*op, "forget_bias", forget_bias);
  return op;
}

// Training

OperatorDef* NetUtil::AddAccuracyOp(const std::string& pred,
                                    const std::string& label,
                                    const std::string& accuracy, int top_k) {
  auto op = AddOp("Accuracy", {pred, label}, {accuracy});
  if (top_k) {
    net_add_arg(*op, "top_k", top_k);
  }
  return op;
}

OperatorDef* NetUtil::AddLabelCrossEntropyOp(const std::string& pred,
                                             const std::string& label,
                                             const std::string& xent) {
  return AddOp("LabelCrossEntropy", {pred, label}, {xent});
}

OperatorDef* NetUtil::AddAveragedLossOp(const std::string& input,
                                        const std::string& loss) {
  return AddOp("AveragedLoss", {input}, {loss});
}

OperatorDef* NetUtil::AddDiagonalOp(const std::string& input,
                                    const std::string& diagonal,
                                    const std::vector<int>& offset) {
  auto op = AddOp("Diagonal", {input}, {diagonal});
  net_add_arg(*op, "offset", offset);
  return op;
}

OperatorDef* NetUtil::AddBackMeanOp(const std::string& input,
                                    const std::string& mean, int count) {
  auto op = AddOp("BackMean", {input}, {mean});
  net_add_arg(*op, "count", count);
  return op;
}

OperatorDef* NetUtil::AddSquaredL2Op(const std::string& input,
                                     const std::string& l2) {
  return AddOp("SquaredL2", {input}, {l2});
}

OperatorDef* NetUtil::AddSquaredL2ChannelOp(const std::string& input,
                                            const std::string& l2,
                                            int channel) {
  auto op = AddOp("SquaredL2Channel", {input}, {l2});
  net_add_arg(*op, "channel", channel);
  return op;
}

OperatorDef* NetUtil::AddMeanStdevOp(const std::string& input,
                                     const std::string& mean,
                                     const std::string& scale) {
  return AddOp("MeanStdev", {input}, {mean, scale});
}

OperatorDef* NetUtil::AddAffineScaleOp(const std::string& input,
                                       const std::string& mean,
                                       const std::string& scale,
                                       const std::string& transformed,
                                       bool inverse) {
  auto op = AddOp("AffineScale", {input, mean, scale}, {transformed});
  net_add_arg(*op, "inverse", inverse);
  return op;
}

OperatorDef* NetUtil::AddSliceOp(
    const std::string& input, const std::string& output,
    const std::vector<std::pair<int, int>>& ranges) {
  auto op = AddOp("Slice", {input}, {output});
  auto starts = net_add_arg(*op, "starts");
  auto ends = net_add_arg(*op, "ends");
  for (auto r : ranges) {
    starts->add_ints(r.first);
    ends->add_ints(r.second);
  }
  return op;
}

OperatorDef* NetUtil::AddReshapeOp(const std::string& input,
                                   const std::string& output,
                                   const std::vector<int>& shape) {
  auto op = AddOp("Reshape", {input}, {output, "_"});
  net_add_arg(*op, "shape", shape);
  return op;
}

OperatorDef* NetUtil::AddWeightedSumOp(const std::vector<std::string>& inputs,
                                       const std::string& sum) {
  return AddOp("WeightedSum", inputs, {sum});
}

OperatorDef* NetUtil::AddCheckpointOp(const std::vector<std::string>& inputs,
                                      int every, const std::string& db_type,
                                      const std::string& db) {
  auto op = AddOp("Checkpoint", inputs, {});
  net_add_arg(*op, "every", every);
  net_add_arg(*op, "db_type", db_type);
  net_add_arg(*op, "db", db);
  return op;
}

OperatorDef* NetUtil::AddMomentumSgdOp(const std::string& param,
                                       const std::string& moment,
                                       const std::string& grad,
                                       const std::string& lr) {
  return AddOp("MomentumSGDUpdate", {grad, moment, lr, param},
               {grad, moment, param});
}

OperatorDef* NetUtil::AddAdagradOp(const std::string& param,
                                   const std::string& moment,
                                   const std::string& grad,
                                   const std::string& lr) {
  return AddOp("Adagrad", {param, moment, grad, lr}, {param, moment});
}

OperatorDef* NetUtil::AddAdamOp(const std::string& param,
                                const std::vector<std::string>& moments,
                                const std::string& grad, const std::string& lr,
                                const std::string& iter) {
  auto op = AddOp("Adam", {param}, {param});
  for (auto& moment : moments) {
    op->add_input(moment);
  }
  op->add_input(grad);
  op->add_input(lr);
  op->add_input(iter);
  for (auto& moment : moments) {
    op->add_output(moment);
  }
  return op;
}

OperatorDef* NetUtil::AddRmsPropOp(const std::string& grad,
                                   const std::string& meansq,
                                   const std::string& mom,
                                   const std::string& lr, float decay,
                                   float momentum, float epsilon) {
  auto op = AddOp("RmsProp", {grad, meansq, mom, lr}, {grad, meansq, mom});
  net_add_arg(*op, "decay", decay);
  net_add_arg(*op, "momentum", momentum);
  net_add_arg(*op, "epsilon", epsilon);
  return op;
}

OperatorDef* NetUtil::AddScaleOp(const std::string& input,
                                 const std::string& output, float scale) {
  auto op = AddOp("Scale", {input}, {output});
  net_add_arg(*op, "scale", scale);
  return op;
}

OperatorDef* NetUtil::AddClipOp(const std::string& input,
                                const std::string& output, float min,
                                float max) {
  auto op = AddOp("Clip", {input}, {output});
  net_add_arg(*op, "min", min);
  net_add_arg(*op, "max", max);
  return op;
}

OperatorDef* NetUtil::AddCastOp(const std::string& input,
                                const std::string& output,
                                TensorProto::DataType type) {
  auto op = AddOp("Cast", {input}, {output});
  net_add_arg(*op, "to", type);
  return op;
}

OperatorDef* NetUtil::AddStopGradientOp(const std::string& param) {
  return AddOp("StopGradient", {param}, {param});
}

OperatorDef* NetUtil::AddIterOp(const std::string& iter) {
  return AddOp("Iter", {iter}, {iter});
}

OperatorDef* NetUtil::AddAtomicIterOp(const std::string& mutex,
                                      const std::string& iter) {
  return AddOp("AtomicIter", {mutex, iter}, {iter});
}

OperatorDef* NetUtil::AddLearningRateOp(const std::string& iter,
                                        const std::string& rate,
                                        float base_rate, float gamma) {
  auto op = AddOp("LearningRate", {iter}, {rate});
  net_add_arg(*op, "policy", "step");
  net_add_arg(*op, "stepsize", 1);
  net_add_arg(*op, "base_lr", -base_rate);
  net_add_arg(*op, "gamma", gamma);
  return op;
}

// RNN

OperatorDef* NetUtil::AddRecurrentNetworkOp(const std::string& seq_lengths,
                                            const std::string& hidden_init,
                                            const std::string& cell_init,
                                            const std::string& scope,
                                            const std::string& hidden_output,
                                            const std::string& cell_state,
                                            bool force_cpu) {
  NetDef forward_model;
  NetUtil forward(forward_model);
  forward.SetName(scope);
  forward.SetType("rnn");
  forward.AddInput("input_t");
  forward.AddInput("timestep");
  forward.AddInput(scope + "/hidden_t_prev");
  forward.AddInput(scope + "/cell_t_prev");
  forward.AddInput(scope + "/gates_t_w");
  forward.AddInput(scope + "/gates_t_b");
  auto fc = forward.AddFcOp(scope + "/hidden_t_prev", scope + "/gates_t_w",
                            scope + "/gates_t_b", scope + "/gates_t", 2);
  fc->set_engine("CUDNN");
  auto sum =
      forward.AddSumOp({scope + "/gates_t", "input_t"}, scope + "/gates_t");
  forward.AddInput(seq_lengths);
  auto lstm =
      forward.AddLSTMUnitOp({scope + "/hidden_t_prev", scope + "/cell_t_prev",
                             scope + "/gates_t", seq_lengths, "timestep"},
                            {scope + "/hidden_t", scope + "/cell_t"});
  forward.AddOutput(scope + "/hidden_t");
  forward.AddOutput(scope + "/cell_t");
#ifdef WITH_CUDA
  if (!force_cpu) {
    fc->mutable_device_option()->set_device_type(CUDA);
    sum->mutable_device_option()->set_device_type(CUDA);
    lstm->mutable_device_option()->set_device_type(CUDA);
    forward.SetDeviceCUDA();
  }
#endif

  NetDef backward_model;
  NetUtil backward(backward_model);
  backward.SetName("RecurrentBackwardStep");
  backward.SetType("simple");
  backward.AddGradientOp(*lstm);
  auto grad = backward.AddGradientOp(*fc);
  grad->set_output(2, scope + "/hidden_t_prev_grad_split");
  backward.AddSumOp(
      {scope + "/hidden_t_prev_grad", scope + "/hidden_t_prev_grad_split"},
      scope + "/hidden_t_prev_grad");
  backward.AddInput(scope + "/gates_t");
  backward.AddInput(scope + "/hidden_t_grad");
  backward.AddInput(scope + "/cell_t_grad");
  backward.AddInput("input_t");
  backward.AddInput("timestep");
  backward.AddInput(scope + "/hidden_t_prev");
  backward.AddInput(scope + "/cell_t_prev");
  backward.AddInput(scope + "/gates_t_w");
  backward.AddInput(scope + "/gates_t_b");
  backward.AddInput(seq_lengths);
  backward.AddInput(scope + "/hidden_t");
  backward.AddInput(scope + "/cell_t");
  if (!force_cpu) {
    backward.SetDeviceCUDA();
  }

  auto op =
      AddOp("RecurrentNetwork",
            {scope + "/i2h", hidden_init, cell_init, scope + "/gates_t_w",
             scope + "/gates_t_b", seq_lengths},
            {scope + "/hidden_t_all", hidden_output, scope + "/cell_t_all",
             cell_state, scope + "/step_workspaces"});
  net_add_arg(*op, "outputs_with_grads", std::vector<int>{0});
  net_add_arg(*op, "link_internal",
              std::vector<std::string>{
                  scope + "/hidden_t_prev", scope + "/hidden_t",
                  scope + "/cell_t_prev", scope + "/cell_t", "input_t"});
  net_add_arg(*op, "alias_dst",
              std::vector<std::string>{scope + "/hidden_t_all", hidden_output,
                                       scope + "/cell_t_all", cell_state});
  net_add_arg(*op, "recompute_blobs_on_backward");
  net_add_arg(*op, "timestep", "timestep");
  net_add_arg(*op, "backward_link_external",
              std::vector<std::string>{
                  scope + "/" + scope + "/hidden_t_prev_states_grad",
                  scope + "/" + scope + "/hidden_t_prev_states_grad",
                  scope + "/" + scope + "/cell_t_prev_states_grad",
                  scope + "/" + scope + "/cell_t_prev_states_grad",
                  scope + "/i2h_grad"});
  net_add_arg(*op, "link_external",
              std::vector<std::string>{
                  scope + "/" + scope + "/hidden_t_prev_states",
                  scope + "/" + scope + "/hidden_t_prev_states",
                  scope + "/" + scope + "/cell_t_prev_states",
                  scope + "/" + scope + "/cell_t_prev_states", scope + "/i2h"});
  net_add_arg(*op, "link_offset", std::vector<int>{0, 1, 0, 1, 0});
  net_add_arg(*op, "alias_offset", std::vector<int>{1, -1, 1, -1});
  net_add_arg(
      *op, "recurrent_states",
      std::vector<std::string>{scope + "/" + scope + "/hidden_t_prev_states",
                               scope + "/" + scope + "/cell_t_prev_states"});
  net_add_arg(*op, "backward_link_offset", std::vector<int>{1, 0, 1, 0, 0});
  net_add_arg(*op, "param_grads",
              std::vector<std::string>{scope + "/gates_t_w_grad",
                                       scope + "/gates_t_b_grad"});
  net_add_arg(*op, "backward_link_internal",
              std::vector<std::string>{
                  scope + "/hidden_t_grad", scope + "/hidden_t_prev_grad",
                  scope + "/cell_t_grad", scope + "/cell_t_prev_grad",
                  scope + "/gates_t_grad"});
  net_add_arg(*op, "param", std::vector<int>{3, 4});
  net_add_arg(*op, "step_net", forward.Proto());
  net_add_arg(*op, "backward_step_net", backward.Proto());
  net_add_arg(
      *op, "alias_src",
      std::vector<std::string>{scope + "/" + scope + "/hidden_t_prev_states",
                               scope + "/" + scope + "/hidden_t_prev_states",
                               scope + "/" + scope + "/cell_t_prev_states",
                               scope + "/" + scope + "/cell_t_prev_states"});
  net_add_arg(*op, "initial_recurrent_state_ids", std::vector<int>{1, 2});
  return op;
}

}  // namespace caffe2
