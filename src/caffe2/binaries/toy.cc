#include <caffe2/core/init.h>
#include <caffe2/core/operator.h>
#include <caffe2/core/operator_gradient.h>

namespace caffe2 {

void print(const Blob *blob, const std::string &name) {
  auto tensor = blob->Get<TensorCPU>();
  const auto &data = tensor.data<float>();
  std::cout << name << "(" << tensor.dims()
            << "): " << std::vector<float>(data, data + tensor.size())
            << std::endl;
}

void run() {
  std::cout << std::endl;
  std::cout << "## Caffe2 Toy Regression Tutorial ##" << std::endl;
  std::cout << "https://caffe2.ai/docs/tutorial-toy-regression.html"
            << std::endl;
  std::cout << std::endl;

  // >>> from caffe2.python import core, cnn, net_drawer, workspace, visualize
  Workspace workspace;

  // >>> init_net = core.Net("init")
  NetDef initModel;
  initModel.set_name("init");

  // >>> W_gt = init_net.GivenTensorFill([], "W_gt", shape=[1, 2],
  // values=[2.0, 1.5])
  {
    auto op = initModel.add_op();
    op->set_type("GivenTensorFill");
    auto arg1 = op->add_arg();
    arg1->set_name("shape");
    arg1->add_ints(1);
    arg1->add_ints(2);
    auto arg2 = op->add_arg();
    arg2->set_name("values");
    arg2->add_floats(2.0);
    arg2->add_floats(1.5);
    op->add_output("W_gt");
  }

  // >>> B_gt = init_net.GivenTensorFill([], "B_gt", shape=[1], values=[0.5])
  {
    auto op = initModel.add_op();
    op->set_type("GivenTensorFill");
    auto arg1 = op->add_arg();
    arg1->set_name("shape");
    arg1->add_ints(1);
    auto arg2 = op->add_arg();
    arg2->set_name("values");
    arg2->add_floats(0.5);
    op->add_output("B_gt");
  }

  // >>> ONE = init_net.ConstantFill([], "ONE", shape=[1], value=1.)
  {
    auto op = initModel.add_op();
    op->set_type("ConstantFill");
    auto arg1 = op->add_arg();
    arg1->set_name("shape");
    arg1->add_ints(1);
    auto arg2 = op->add_arg();
    arg2->set_name("value");
    arg2->set_f(1.0);
    op->add_output("ONE");
  }

  // >>> ITER = init_net.ConstantFill([], "ITER", shape=[1], value=0,
  // dtype=core.DataType.INT32)
  {
    auto op = initModel.add_op();
    op->set_type("ConstantFill");
    auto arg1 = op->add_arg();
    arg1->set_name("shape");
    arg1->add_ints(1);
    auto arg2 = op->add_arg();
    arg2->set_name("value");
    arg2->set_i(0);
    auto arg3 = op->add_arg();
    arg3->set_name("dtype");
    arg3->set_i(TensorProto_DataType_INT32);
    op->add_output("ITER");
  }

  // >>> W = init_net.UniformFill([], "W", shape=[1, 2], min=-1., max=1.)
  {
    auto op = initModel.add_op();
    op->set_type("UniformFill");
    auto arg1 = op->add_arg();
    arg1->set_name("shape");
    arg1->add_ints(1);
    arg1->add_ints(2);
    auto arg2 = op->add_arg();
    arg2->set_name("min");
    arg2->set_f(-1);
    auto arg3 = op->add_arg();
    arg3->set_name("max");
    arg3->set_f(1);
    op->add_output("W");
  }

  // >>> B = init_net.ConstantFill([], "B", shape=[1], value=0.0)
  {
    auto op = initModel.add_op();
    op->set_type("ConstantFill");
    auto arg1 = op->add_arg();
    arg1->set_name("shape");
    arg1->add_ints(1);
    auto arg2 = op->add_arg();
    arg2->set_name("value");
    arg2->set_f(0);
    op->add_output("B");
  }

  // print(initModel);

  // >>> train_net = core.Net("train")
  NetDef trainModel;
  trainModel.set_name("train");

  // >>> X = train_net.GaussianFill([], "X", shape=[64, 2], mean=0.0, std=1.0,
  // run_once=0)
  {
    auto op = trainModel.add_op();
    op->set_type("GaussianFill");
    auto arg1 = op->add_arg();
    arg1->set_name("shape");
    arg1->add_ints(64);
    arg1->add_ints(2);
    auto arg2 = op->add_arg();
    arg2->set_name("mean");
    arg2->set_f(0);
    auto arg3 = op->add_arg();
    arg3->set_name("std");
    arg3->set_f(1);
    auto arg4 = op->add_arg();
    arg4->set_name("run_once");
    arg4->set_i(0);
    op->add_output("X");
  }

  // >>> Y_gt = X.FC([W_gt, B_gt], "Y_gt")
  {
    auto op = trainModel.add_op();
    op->set_type("FC");
    op->add_input("X");
    op->add_input("W_gt");
    op->add_input("B_gt");
    op->add_output("Y_gt");
  }

  // >>> noise = train_net.GaussianFill([], "noise", shape=[64, 1], mean=0.0,
  // std=1.0, run_once=0)
  {
    auto op = trainModel.add_op();
    op->set_type("GaussianFill");
    auto arg1 = op->add_arg();
    arg1->set_name("shape");
    arg1->add_ints(64);
    arg1->add_ints(1);
    auto arg2 = op->add_arg();
    arg2->set_name("mean");
    arg2->set_f(0);
    auto arg3 = op->add_arg();
    arg3->set_name("std");
    arg3->set_f(1);
    auto arg4 = op->add_arg();
    arg4->set_name("run_once");
    arg4->set_i(0);
    op->add_output("noise");
  }

  // >>> Y_noise = Y_gt.Add(noise, "Y_noise")
  {
    auto op = trainModel.add_op();
    op->set_type("Add");
    op->add_input("Y_gt");
    op->add_input("noise");
    op->add_output("Y_noise");
  }

  // >>> Y_noise = Y_noise.StopGradient([], "Y_noise")
  {
    auto op = trainModel.add_op();
    op->set_type("StopGradient");
    op->add_input("Y_noise");
    op->add_output("Y_noise");
  }

  std::vector<OperatorDef *> gradient_ops;

  // >>> Y_pred = X.FC([W, B], "Y_pred")
  {
    auto op = trainModel.add_op();
    op->set_type("FC");
    op->add_input("X");
    op->add_input("W");
    op->add_input("B");
    op->add_output("Y_pred");
    gradient_ops.push_back(op);
  }

  // >>> dist = train_net.SquaredL2Distance([Y_noise, Y_pred], "dist")
  {
    auto op = trainModel.add_op();
    op->set_type("SquaredL2Distance");
    op->add_input("Y_noise");
    op->add_input("Y_pred");
    op->add_output("dist");
    gradient_ops.push_back(op);
  }

  // >>> loss = dist.AveragedLoss([], ["loss"])
  {
    auto op = trainModel.add_op();
    op->set_type("AveragedLoss");
    op->add_input("dist");
    op->add_output("loss");
    gradient_ops.push_back(op);
  }

  // >>> gradient_map = train_net.AddGradientOperators([loss])
  {
    auto op = trainModel.add_op();
    op->set_type("ConstantFill");
    auto arg = op->add_arg();
    arg->set_name("value");
    arg->set_f(1.0);
    op->add_input("loss");
    op->add_output("loss_grad");
    op->set_is_gradient_op(true);
  }
  std::reverse(gradient_ops.begin(), gradient_ops.end());
  for (auto op : gradient_ops) {
    vector<GradientWrapper> output(op->output_size());
    for (auto i = 0; i < output.size(); i++) {
      output[i].dense_ = op->output(i) + "_grad";
    }
    GradientOpsMeta meta = GetGradientForOp(*op, output);
    auto grad = trainModel.add_op();
    grad->CopyFrom(meta.ops_[0]);
    grad->set_is_gradient_op(true);
  }

  // >>> train_net.Iter(ITER, ITER)
  {
    auto op = trainModel.add_op();
    op->set_type("Iter");
    op->add_input("ITER");
    op->add_output("ITER");
  }

  // >>> LR = train_net.LearningRate(ITER, "LR", base_lr=-0.1, policy="step",
  // stepsize=20, gamma=0.9)
  {
    auto op = trainModel.add_op();
    op->set_type("LearningRate");
    auto arg1 = op->add_arg();
    arg1->set_name("base_lr");
    arg1->set_f(-0.1);
    auto arg2 = op->add_arg();
    arg2->set_name("policy");
    arg2->set_s("step");
    auto arg3 = op->add_arg();
    arg3->set_name("stepsize");
    arg3->set_i(20);
    auto arg4 = op->add_arg();
    arg4->set_name("gamma");
    arg4->set_f(0.9);
    op->add_input("ITER");
    op->add_output("LR");
  }

  // >>> train_net.WeightedSum([W, ONE, gradient_map[W], LR], W)
  {
    auto op = trainModel.add_op();
    op->set_type("WeightedSum");
    op->add_input("W");
    op->add_input("ONE");
    op->add_input("W_grad");
    op->add_input("LR");
    op->add_output("W");
  }

  // >>> train_net.WeightedSum([B, ONE, gradient_map[B], LR], B)
  {
    auto op = trainModel.add_op();
    op->set_type("WeightedSum");
    op->add_input("B");
    op->add_input("ONE");
    op->add_input("B_grad");
    op->add_input("LR");
    op->add_output("B");
  }

  // print(trainModel);

  // >>> workspace.RunNetOnce(init_net)
  auto initNet = CreateNet(initModel, &workspace);
  initNet->Run();

  // >>> workspace.CreateNet(train_net)
  auto trainNet = CreateNet(trainModel, &workspace);

  // >>> print("Before training, W is: {}".format(workspace.FetchBlob("W")))
  print(workspace.GetBlob("W"), "W before");

  // >>> print("Before training, B is: {}".format(workspace.FetchBlob("B")))
  print(workspace.GetBlob("B"), "B before");

  // >>> for i in range(100):
  for (auto i = 1; i <= 100; i++) {
    // >>> workspace.RunNet(train_net.Proto().name)
    trainNet->Run();

    if (i % 10 == 0) {
      float w = workspace.GetBlob("W")->Get<TensorCPU>().data<float>()[0];
      float b = workspace.GetBlob("B")->Get<TensorCPU>().data<float>()[0];
      float loss = workspace.GetBlob("loss")->Get<TensorCPU>().data<float>()[0];
      std::cout << "step: " << i << " W: " << w << " B: " << b
                << " loss: " << loss << std::endl;
    }
  }

  // >>> print("After training, W is: {}".format(workspace.FetchBlob("W")))
  print(workspace.GetBlob("W"), "W after");

  // >>> print("After training, B is: {}".format(workspace.FetchBlob("B")))
  print(workspace.GetBlob("B"), "B after");

  // >>> print("Ground truth W is: {}".format(workspace.FetchBlob("W_gt")))
  print(workspace.GetBlob("W_gt"), "W ground truth");

  // >>> print("Ground truth B is: {}".format(workspace.FetchBlob("B_gt")))
  print(workspace.GetBlob("B_gt"), "B ground truth");
}

}  // namespace caffe2

int main(int argc, char **argv) {
  caffe2::GlobalInit(&argc, &argv);
  caffe2::run();
  google::protobuf::ShutdownProtobufLibrary();
  return 0;
}
