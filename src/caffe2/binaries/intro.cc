#include <caffe2/core/init.h>
#include <caffe2/core/operator.h>
#include <caffe2/core/operator_gradient.h>
#include <caffe2/util/blob.h>
#include <caffe2/util/model.h>
#include <caffe2/util/net.h>

namespace caffe2 {

void print(const Blob* blob, const std::string& name) {
  auto tensor = blob->Get<TensorCPU>();
  const auto& data = tensor.data<float>();
  std::cout << name << "(" << tensor.dims()
            << "): " << std::vector<float>(data, data + tensor.size())
            << std::endl;
}

void run() {
  std::cout << std::endl;
  std::cout << "## Caffe2 Intro Tutorial ##" << std::endl;
  std::cout << "https://caffe2.ai/docs/intro-tutorial.html" << std::endl;
  std::cout << std::endl;

  // >>> from caffe2.python import workspace, model_helper
  // >>> import numpy as np
  Workspace workspace;

  // >>> x = np.random.rand(4, 3, 2)
  std::vector<float> x(4 * 3 * 2);
  for (auto& v : x) {
    v = (float)rand() / RAND_MAX;
  }

  // >>> print(x)
  std::cout << x << std::endl;

  // >>> workspace.FeedBlob("my_x", x)
  {
    auto tensor = workspace.CreateBlob("my_x")->GetMutable<TensorCPU>();
    auto value = TensorCPU({4, 3, 2}, x, NULL);
    tensor->ResizeLike(value);
    tensor->ShareData(value);
  }

  // >>> x2 = workspace.FetchBlob("my_x")
  // >>> print(x2)
  {
    const auto blob = workspace.GetBlob("my_x");
    print(blob, "my_x");
  }

  // >>> data = np.random.rand(16, 100).astype(np.float32)
  std::vector<float> data(16 * 100);
  for (auto& v : data) {
    v = (float)rand() / RAND_MAX;
  }

  // >>> label = (np.random.rand(16) * 10).astype(np.int32)
  std::vector<int> label(16);
  for (auto& v : label) {
    v = 10 * rand() / RAND_MAX;
  }

  // >>> workspace.FeedBlob("data", data)
  {
    auto tensor = workspace.CreateBlob("data")->GetMutable<TensorCPU>();
    auto value = TensorCPU({16, 100}, data, NULL);
    tensor->ResizeLike(value);
    tensor->ShareData(value);
  }

  // >>> workspace.FeedBlob("label", label)
  {
    auto tensor = workspace.CreateBlob("label")->GetMutable<TensorCPU>();
    auto value = TensorCPU({16}, label, NULL);
    tensor->ResizeLike(value);
    tensor->ShareData(value);
  }

  // >>> m = model_helper.ModelHelper(name="my first net")
  NetDef initModel;
  NetDef predictModel;
  ModelUtil model_helper(initModel,predictModel,"my first net");

  // >>> weight = m.param_initModel.XavierFill([], 'fc_w', shape=[10, 100])
  // >>> bias = m.param_initModel.ConstantFill([], 'fc_b', shape=[10, ])
  // >>> fc_1 = m.net.FC(["data", "fc_w", "fc_b"], "fc1")
  model_helper.predict.AddInput("data");
  model_helper.AddFcOps("data","fc1",100,10);
  
  // >>> pred = m.net.Sigmoid(fc_1, "pred")
  model_helper.AddSigmoidOp("fc1","pred");

  // >>> [softmax, loss] = m.net.SoftmaxWithLoss([pred, "label"], ["softmax",
  // "loss"])
  model_helper.predict.AddInput("pred");
  model_helper.predict.AddInput("label");
  model_helper.AddSoftmaxWithLossOp({"pred","label"},{"softmax","loss"});

  // >>> m.AddGradientOperators([loss])
  model_helper.AddGradientOps();

  // >>> print(str(m.net.Proto()))
  model_helper.predict.Print();

  // >>> print(str(m.param_init_net.Proto()))
  model_helper.init.Print();

  // >>> workspace.RunNetOnce(m.param_init_net)
  CAFFE_ENFORCE(workspace.RunNetOnce(initModel));

  // >>> workspace.CreateNet(m.net)
  CAFFE_ENFORCE(workspace.CreateNet(predictModel));

  // >>> for j in range(0, 100):
  for (auto i = 0; i < 100; i++) {
    // >>> data = np.random.rand(16, 100).astype(np.float32)
    std::vector<float> data(16 * 100);
    for (auto& v : data) {
      v = (float)rand() / RAND_MAX;
    }

    // >>> label = (np.random.rand(16) * 10).astype(np.int32)
    std::vector<int> label(16);
    for (auto& v : label) {
      v = 10 * rand() / RAND_MAX;
    }

    // >>> workspace.FeedBlob("data", data)
    {
      auto tensor = workspace.GetBlob("data")->GetMutable<TensorCPU>();
      auto value = TensorCPU({16, 100}, data, NULL);
      tensor->ShareData(value);
    }

    // >>> workspace.FeedBlob("label", label)
    {
      auto tensor = workspace.GetBlob("label")->GetMutable<TensorCPU>();
      auto value = TensorCPU({16}, label, NULL);
      tensor->ShareData(value);
    }

    // >>> workspace.RunNet(m.name, 10)   # run for 10 times
    for (auto j = 0; j < 10; j++) {
      CAFFE_ENFORCE(workspace.RunNet(predictModel.name()));
      // std::cout << "step: " << i << " loss: ";
      // print(*(workspace.GetBlob("loss")));
      // std::cout << std::endl;
    }
  }

  std::cout << std::endl;

  // >>> print(workspace.FetchBlob("softmax"))
  print(workspace.GetBlob("softmax"), "softmax");

  std::cout << std::endl;

  // >>> print(workspace.FetchBlob("loss"))
  print(workspace.GetBlob("loss"), "loss");
}

}  // namespace caffe2

int main(int argc, char** argv) {
  caffe2::GlobalInit(&argc, &argv);
  caffe2::run();
  google::protobuf::ShutdownProtobufLibrary();
  return 0;
}
