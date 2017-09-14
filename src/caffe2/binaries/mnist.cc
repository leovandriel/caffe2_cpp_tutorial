#include "caffe2/core/init.h"
#include "caffe2/util/net.h"

#include "caffe2/util/window.h"

#ifdef WITH_CUDA
#include "caffe2/core/context_gpu.h"
#endif

CAFFE2_DEFINE_string(train_db, "res/mnist-train-nchw-leveldb",
                     "The given path to the training leveldb.");
CAFFE2_DEFINE_string(test_db, "res/mnist-test-nchw-leveldb",
                     "The given path to the testing leveldb.");
CAFFE2_DEFINE_int(train_runs, 100, "The of training runs.");
CAFFE2_DEFINE_int(test_runs, 50, "The of test runs.");
CAFFE2_DEFINE_bool(force_cpu, false, "Only use CPU, no CUDA.");
CAFFE2_DEFINE_bool(display, false, "Display graphical training info.");

namespace caffe2 {

// >> def AddInput(model, batch_size, db, db_type):
void AddInput(NetUtil &init, NetUtil &predict, int batch_size,
              const std::string &db, const std::string &db_type) {
  // Setup database connection
  init.AddCreateDbOp("dbreader", db_type, db);
  predict.AddInput("dbreader");

  // >>> data_uint8, label = model.TensorProtosDBInput([], ["data_uint8",
  // "label"], batch_size=batch_size, db=db, db_type=db_type)
  predict.AddTensorProtosDbInputOp("dbreader", "data_uint8", "label",
                                   batch_size);

  // >>> data = model.Cast(data_uint8, "data", to=core.DataType.FLOAT)
  predict.AddCastOp("data_uint8", "data", TensorProto_DataType_FLOAT);

  // >>> data = model.Scale(data, data, scale=float(1./256))
  predict.AddScaleOp("data", "data", 1.f / 256);

  // >>> data = model.StopGradient(data, data)
  predict.AddStopGradientOp("data");
}

// def AddLeNetModel(model, data):
void AddLeNetModel(NetUtil &init, NetUtil &predict, bool training) {
  // >>> conv1 = brew.conv(model, data, 'conv1', dim_in=1, dim_out=20, kernel=5)
  predict.AddConvOp("data", "conv1_w", "conv1_b", "conv1", 1, 0, 5);
  predict.AddInput("conv1_w");
  predict.AddInput("conv1_b");
  if (training) {
    init.AddXavierFillOp({20, 1, 5, 5}, "conv1_w");
    init.AddConstantFillOp({20}, "conv1_b");
  }

  // >>> pool1 = brew.max_pool(model, conv1, 'pool1', kernel=2, stride=2)
  predict.AddMaxPoolOp("conv1", "pool1", 2, 0, 2);

  // >>> conv2 = brew.conv(model, pool1, 'conv2', dim_in=20, dim_out=50,
  // kernel=5)
  predict.AddConvOp("pool1", "conv2_w", "conv2_b", "conv2", 1, 0, 5);
  predict.AddInput("conv2_w");
  predict.AddInput("conv2_b");
  if (training) {
    init.AddXavierFillOp({50, 20, 5, 5}, "conv2_w");
    init.AddConstantFillOp({50}, "conv2_b");
  }

  // >>> pool2 = brew.max_pool(model, conv2, 'pool2', kernel=2, stride=2)
  predict.AddMaxPoolOp("conv2", "pool2", 2, 0, 2);

  // >>> fc3 = brew.fc(model, pool2, 'fc3', dim_in=50 * 4 * 4, dim_out=500)
  predict.AddFcOp("pool2", "fc3_w", "fc3_b", "fc3");
  predict.AddInput("fc3_w");
  predict.AddInput("fc3_b");
  if (training) {
    init.AddXavierFillOp({500, 800}, "fc3_w");
    init.AddConstantFillOp({500}, "fc3_b");
  }

  // >>> fc3 = brew.relu(model, fc3, fc3)
  predict.AddReluOp("fc3", "fc3");

  // >>> pred = brew.fc(model, fc3, 'pred', 500, 10)
  predict.AddFcOp("fc3", "pred_w", "pred_b", "pred");
  predict.AddInput("pred_w");
  predict.AddInput("pred_b");
  if (training) {
    init.AddXavierFillOp({10, 500}, "pred_w");
    init.AddConstantFillOp({10}, "pred_b");
  }

  // >>> softmax = brew.softmax(model, pred, 'softmax')
  predict.AddSoftmaxOp("pred", "softmax");
}

// def AddAccuracy(model, softmax, label):
void AddAccuracy(NetUtil &init, NetUtil &predict) {
  // >>> accuracy = model.Accuracy([softmax, label], "accuracy")
  predict.AddAccuracyOp("softmax", "label", "accuracy");

  if (FLAGS_display) {
    NetUtil(predict).AddTimePlotOp("accuracy", {"accuracy"});
  }

  // Moved ITER to AddAccuracy function, so it's also available on test runs
  init.AddConstantFillOp({1}, (int64_t)0, "ITER")
      ->mutable_device_option()
      ->set_device_type(CPU);
  predict.AddInput("ITER");

  // >>> ITER = model.Iter("iter")
  predict.AddIterOp("ITER");
}

// >>> def AddTrainingOperators(model, softmax, label):
void AddTrainingOperators(NetUtil &init, NetUtil &predict,
                          std::vector<string> params) {
  // >>> xent = model.LabelCrossEntropy([softmax, label], 'xent')
  predict.AddLabelCrossEntropyOp("softmax", "label", "xent");

  // >>> loss = model.AveragedLoss(xent, "loss")
  predict.AddAveragedLossOp("xent", "loss");

  if (FLAGS_display) {
    NetUtil(predict).AddShowWorstOp("softmax", "label", "data");
    NetUtil(predict).AddTimePlotOp("loss", {"loss"});
  }

  // >>> AddAccuracy(model, softmax, label)
  AddAccuracy(init, predict);

  // >>> model.AddGradientOperators([loss])
  predict.AddConstantFillWithOp(1.f, "loss", "loss_grad");
  predict.AddGradientOps();

  // >>> LR = model.LearningRate(ITER, "LR", base_lr=-0.1, policy="step",
  // stepsize=1, gamma=0.999 )
  predict.AddLearningRateOp("ITER", "LR", 0.1);

  // >>> ONE = model.param_init_net.ConstantFill([], "ONE", shape=[1],
  // value=1.0)
  init.AddConstantFillOp({1}, 1.f, "ONE");
  predict.AddInput("ONE");

  // >>> for param in model.params:
  for (auto param : params) {
    // >>> param_grad = model.param_to_grad[param]
    // >>> model.WeightedSum([param, ONE, param_grad, LR], param)
    predict.AddWeightedSumOp({param, "ONE", param + "_grad", "LR"}, param);
  }

  return;  // Checkpoint causes problems on subsequent runs

  // >>> model.Checkpoint([ITER] + model.params, [],
  std::vector<std::string> inputs({"ITER"});
  inputs.insert(inputs.end(), params.begin(), params.end());
  predict.AddCheckpointOp(inputs, 20, "leveldb",
                          "mnist_lenet_checkpoint_%05d.leveldb");
}

// >>> def AddBookkeepingOperators(model):
void AddBookkeepingOperators(NetUtil &init, NetUtil &predict,
                             std::vector<string> params) {
  // >>> model.Print('accuracy', [], to_file=1)
  predict.AddPrintOp("accuracy", true);

  // >>> model.Print('loss', [], to_file=1)
  predict.AddPrintOp("loss", true);

  // >>> for param in model.params:
  for (auto param : params) {
    // >>> model.Summarize(param, [], to_file=1)
    predict.AddSummarizeOp(param, true);

    // >>> model.Summarize(model.param_to_grad[param], [], to_file=1)
    predict.AddSummarizeOp(param + "_grad", true);
  }
}

TensorCPU GetTensor(const Blob &blob) {
#ifdef WITH_CUDA
  return TensorCPU(blob.Get<TensorCUDA>());
#else
  return blob.Get<TensorCPU>();
#endif
}

void run() {
  std::cout << std::endl;
  std::cout << "## Caffe2 MNIST Tutorial ##" << std::endl;
  std::cout << "https://caffe2.ai/docs/tutorial-MNIST.html" << std::endl;
  std::cout << std::endl;

  if (!std::ifstream(FLAGS_train_db).good() ||
      !std::ifstream(FLAGS_test_db).good()) {
    std::cerr << "error: MNIST database missing: "
              << (std::ifstream(FLAGS_train_db).good() ? FLAGS_test_db
                                                       : FLAGS_train_db)
              << std::endl;
    std::cerr << "Make sure to first run ./script/download_resource.sh"
              << std::endl;
    return;
  }

  std::cout << "train_db: " << FLAGS_train_db << std::endl;
  std::cout << "test_db: " << FLAGS_test_db << std::endl;
  std::cout << "train_runs: " << FLAGS_train_runs << std::endl;
  std::cout << "test_runs: " << FLAGS_test_runs << std::endl;
  std::cout << "force_cpu: " << (FLAGS_force_cpu ? "true" : "false")
            << std::endl;
  std::cout << "display: " << (FLAGS_display ? "true" : "false") << std::endl;

#ifdef WITH_CUDA
  if (!FLAGS_force_cpu) {
    DeviceOption option;
    option.set_device_type(CUDA);
    new CUDAContext(option);
    std::cout << std::endl << "using CUDA" << std::endl;
  }
#endif

  if (FLAGS_display) {
    superWindow("Caffe2 MNIST Tutorial");
    moveWindow("worst_pos", 0, 0);
    resizeWindow("worst_pos", 260, 260);
    moveWindow("worst_neg", 0, 260);
    resizeWindow("worst_neg", 260, 260);
    moveWindow("accuracy", 260, 0);
    moveWindow("loss", 260, 260);
  }

  // >>> from caffe2.python import core, cnn, net_drawer, workspace, visualize,
  // brew
  // >>> workspace.ResetWorkspace(root_folder)
  Workspace workspace("tmp");

  // >>> train_model = model_helper.ModelHelper(name="mnist_train",
  // arg_scope={"order": "NCHW"})
  NetDef initTrainModel, predictTrainModel;
  NetUtil initTrain(initTrainModel), predictTrain(predictTrainModel);
  initTrain.SetName("mnist_train_init");
  predictTrain.SetName("mnist_train_predict");

  std::vector<string> params({"conv1_w", "conv1_b", "conv2_w", "conv2_b",
                              "fc3_w", "fc3_b", "pred_w", "pred_b"});

  // >>> data, label = AddInput(train_model, batch_size=64,
  // db=os.path.join(data_folder, 'mnist-train-nchw-leveldb'),
  // db_type='leveldb')
  AddInput(initTrain, predictTrain, 64, FLAGS_train_db, "leveldb");

  // >>> softmax = AddLeNetModel(train_model, data)
  AddLeNetModel(initTrain, predictTrain, true);

  // >>> AddTrainingOperators(train_model, softmax, label)
  AddTrainingOperators(initTrain, predictTrain, params);

  // >>> AddBookkeepingOperators(train_model)
  AddBookkeepingOperators(initTrain, predictTrain, params);

  // >>> test_model = model_helper.ModelHelper(name="mnist_test",
  // arg_scope=arg_scope, init_params=False)
  NetDef initTestModel, predictTestModel;
  NetUtil initTest(initTestModel), predictTest(predictTestModel);
  initTest.SetName("mnist_test_init");
  predictTest.SetName("mnist_test_predict");

  // >>> data, label = AddInput(test_model, batch_size=100,
  // db=os.path.join(data_folder, 'mnist-test-nchw-leveldb'), db_type='leveldb')
  AddInput(initTest, predictTest, 100, FLAGS_test_db, "leveldb");

  // >>> softmax = AddLeNetModel(test_model, data)
  AddLeNetModel(initTest, predictTest, false);

  // >>> AddAccuracy(test_model, softmax, label)
  AddAccuracy(initTest, predictTest);

  // >>> deploy_model = model_helper.ModelHelper(name="mnist_deploy",
  // arg_scope=arg_scope, init_params=False)
  NetDef initDeployModel, predictDeployModel;
  NetUtil initDeploy(initDeployModel), predictDeploy(predictDeployModel);
  initDeploy.SetName("mnist_model_init");
  predictDeploy.SetName("mnist_model_predict");

  // >>> AddLeNetModel(deploy_model, "data")
  AddLeNetModel(initDeploy, predictDeploy, false);

#ifdef WITH_CUDA
  if (!FLAGS_force_cpu) {
    initTrainModel.mutable_device_option()->set_device_type(CUDA);
    predictTrainModel.mutable_device_option()->set_device_type(CUDA);
    initTestModel.mutable_device_option()->set_device_type(CUDA);
    predictTestModel.mutable_device_option()->set_device_type(CUDA);
  }
#endif

  std::cout << std::endl;

  // >>> workspace.RunNetOnce(train_model.param_init_net)
  auto initTrainNet = CreateNet(initTrainModel, &workspace);
  initTrainNet->Run();

  // >>> workspace.CreateNet(train_model.net)
  auto predictTrainNet = CreateNet(predictTrainModel, &workspace);

  std::cout << "training.." << std::endl;

  // >>> for i in range(total_iters):
  for (auto i = 1; i <= FLAGS_train_runs; i++) {
    // >>> workspace.RunNet(train_model.net.Proto().name)
    predictTrainNet->Run();

    // >>> accuracy[i] = workspace.FetchBlob('accuracy')
    // >>> loss[i] = workspace.FetchBlob('loss')
    if (i % 10 == 0) {
      auto accuracy =
          GetTensor(*workspace.GetBlob("accuracy")).data<float>()[0];
      auto loss = GetTensor(*workspace.GetBlob("loss")).data<float>()[0];
      std::cout << "step: " << i << " loss: " << loss
                << " accuracy: " << accuracy << std::endl;
    }
  }

  std::cout << std::endl;

  // >>> workspace.RunNetOnce(test_model.param_init_net)
  auto initTestNet = CreateNet(initTestModel, &workspace);
  initTestNet->Run();

  // >>> workspace.CreateNet(test_model.net)
  auto predictTestNet = CreateNet(predictTestModel, &workspace);

  std::cout << "testing.." << std::endl;

  // >>> for i in range(100):
  for (auto i = 1; i <= FLAGS_test_runs; i++) {
    // >>> workspace.RunNet(test_model.net.Proto().name)
    predictTestNet->Run();

    // >>> test_accuracy[i] = workspace.FetchBlob('accuracy')
    if (i % 10 == 0) {
      auto accuracy =
          GetTensor(*workspace.GetBlob("accuracy")).data<float>()[0];
      std::cout << "step: " << i << " accuracy: " << accuracy << std::endl;
    }
  }

  // with open(os.path.join(root_folder, "deploy_net.pbtxt"), 'w') as fid:
  // fid.write(str(deploy_model.net.Proto()))
  std::vector<string> external(initDeployModel.external_input().begin(),
                               initDeployModel.external_input().end());
  for (auto &param : external) {
    auto tensor = GetTensor(*workspace.GetBlob(param));
    auto op = initDeployModel.add_op();
    op->set_type("GivenTensorFill");
    auto arg1 = op->add_arg();
    arg1->set_name("shape");
    for (auto d : tensor.dims()) {
      arg1->add_ints(d);
    }
    auto arg2 = op->add_arg();
    arg2->set_name("values");
    auto data = tensor.data<float>();
    for (auto i = 0; i < tensor.size(); i++) {
      arg2->add_floats(data[i]);
    }
  }
  WriteProtoToTextFile(predictDeployModel, "tmp/mnist_predict_net.pbtxt");
  WriteProtoToBinaryFile(initDeployModel, "tmp/mnist_init_net.pb");
  WriteProtoToBinaryFile(predictDeployModel, "tmp/mnist_predict_net.pb");
}

}  // namespace caffe2

int main(int argc, char **argv) {
  caffe2::GlobalInit(&argc, &argv);
  caffe2::run();
  google::protobuf::ShutdownProtobufLibrary();
  return 0;
}
