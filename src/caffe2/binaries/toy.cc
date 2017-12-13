#include <caffe2/core/init.h>
#include <caffe2/core/operator.h>
#include <caffe2/core/operator_gradient.h>
//model helper
#include <caffe2/util/blob.h>
#include <caffe2/util/model.h>
#include <caffe2/util/net.h>

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
  NetDef initModel,trainModel;
  ModelUtil model_helper(initModel,trainModel,"toy");

  // >>> W_gt = init_net.GivenTensorFill([], "W_gt", shape=[1, 2],
  // values=[2.0, 1.5])
  {
		vector<float> data = {2.0,1.5};
		auto tensor = TensorCPU({1,2},data,NULL);
		model_helper.init.AddGivenTensorFillOp(tensor,"W_gt");
  }

  // >>> B_gt = init_net.GivenTensorFill([], "B_gt", shape=[1], values=[0.5])
  {
	  vector<float> data = {0.5};
	  auto tensor = TensorCPU({1},data,NULL);
	  model_helper.init.AddGivenTensorFillOp(tensor,"B_gt");
  }

  // >>> ONE = init_net.ConstantFill([], "ONE", shape=[1], value=1.)
  model_helper.init.AddConstantFillOp({1},1.0f,"ONE");

  // >>> ITER = init_net.ConstantFill([], "ITER", shape=[1], value=0,
  // dtype=core.DataType.INT32)
  model_helper.init.AddConstantFillOp({1},(int64_t)0,"ITER");

  // >>> W = init_net.UniformFill([], "W", shape=[1, 2], min=-1., max=1.)
  model_helper.init.AddUniformFillOp({1,2},-1,1,"W");

  // >>> B = init_net.ConstantFill([], "B", shape=[1], value=0.0)
  model_helper.init.AddConstantFillOp({1},0.0f,"B");

  // print(initModel);
  model_helper.init.Print();

  // >>> train_net = core.Net("train")

  // >>> X = train_net.GaussianFill([], "X", shape=[64, 2], mean=0.0, std=1.0,
  // run_once=0)
  model_helper.AddGaussianFillOp({64,2},0,1,"X");

  // >>> Y_gt = X.FC([W_gt, B_gt], "Y_gt")
  model_helper.predict.AddInput("W_gt");
  model_helper.predict.AddInput("B_gt");
  model_helper.predict.AddFcOp("X","W_gt","B_gt","Y_gt");

  // >>> noise = train_net.GaussianFill([], "noise", shape=[64, 1], mean=0.0,
  // std=1.0, run_once=0)
  model_helper.AddGaussianFillOp({64,1},0,1,"noise");

  // >>> Y_noise = Y_gt.Add(noise, "Y_noise")
  model_helper.AddAddOp({"Y_gt","noise"},"Y_noise",0,0);

  // >>> Y_noise = Y_noise.StopGradient([], "Y_noise")
  model_helper.AddStopGradientOp("Y_noise");

  // >>> Y_pred = X.FC([W, B], "Y_pred")
  model_helper.predict.AddInput("W");
  model_helper.predict.AddInput("B");
  model_helper.predict.AddFcOp("X","W","B","Y_pred");

  // >>> dist = train_net.SquaredL2Distance([Y_noise, Y_pred], "dist")
  model_helper.AddSquaredL2DistanceOp({"Y_noise","Y_pred"},"dist");

  // >>> loss = dist.AveragedLoss([], ["loss"])
  model_helper.AddAveragedLossOp("dist","loss");

  // >>> gradient_map = train_net.AddGradientOperators([loss])
  model_helper.predict.AddConstantFillWithOp(1.0, "loss", "loss_grad");
  model_helper.predict.AddGradientOps();

  // >>> train_net.Iter(ITER, ITER)
  model_helper.predict.AddInput("ITER");
  model_helper.predict.AddIterOp("ITER");

  // >>> LR = train_net.LearningRate(ITER, "LR", base_lr=-0.1, policy="step",
  // stepsize=20, gamma=0.9)
  model_helper.AddLearningRateOp("ITER","LR",-0.1,0.9, 20);

  // >>> train_net.WeightedSum([W, ONE, gradient_map[W], LR], W)
  model_helper.predict.AddInput("ONE");
  model_helper.AddWeightedSumOp({"W","ONE","W_grad","LR"},"W");

  // >>> train_net.WeightedSum([B, ONE, gradient_map[B], LR], B)
  model_helper.AddWeightedSumOp({"B","ONE","B_grad","LR"},"B");

  // print(trainModel);
  model_helper.predict.Print();

  // >>> workspace.RunNetOnce(init_net)
  CAFFE_ENFORCE(workspace.RunNetOnce(initModel));

  // >>> workspace.CreateNet(train_net)
  CAFFE_ENFORCE(workspace.CreateNet(trainModel));

  // >>> print("Before training, W is: {}".format(workspace.FetchBlob("W")))
  print(workspace.GetBlob("W"), "W before");

  // >>> print("Before training, B is: {}".format(workspace.FetchBlob("B")))
  print(workspace.GetBlob("B"), "B before");

  // >>> for i in range(100):
  for (auto i = 1; i <= 100; i++) {
    // >>> workspace.RunNet(train_net.Proto().name)
    CAFFE_ENFORCE(workspace.RunNet(trainModel.name()));

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
