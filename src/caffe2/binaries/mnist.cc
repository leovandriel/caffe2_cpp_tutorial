#include <caffe2/core/init.h>
#include "caffe2/util/blob.h"
#include "caffe2/util/model.h"
#include "caffe2/util/net.h"

#include "caffe2/util/window.h"

#ifdef WITH_CUDA
#include <caffe2/core/context_gpu.h>
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
void AddInput(ModelUtil &model, int batch_size, const std::string &db,
              const std::string &db_type) {
  // Setup database connection
  model.init.AddCreateDbOp("dbreader", db_type, db);
  model.predict.AddInput("dbreader");

  // >>> data_uint8, label = model.TensorProtosDBInput([], ["data_uint8",
  // "label"], batch_size=batch_size, db=db, db_type=db_type)
  model.predict.AddTensorProtosDbInputOp("dbreader", "data_uint8", "label",
                                         batch_size);

  // >>> data = model.Cast(data_uint8, "data", to=core.DataType.FLOAT)
  model.predict.AddCastOp("data_uint8", "data", TensorProto_DataType_FLOAT);

  // >>> data = model.Scale(data, data, scale=float(1./256))
  model.predict.AddScaleOp("data", "data", 1.f / 256);

  // >>> data = model.StopGradient(data, data)
  model.predict.AddStopGradientOp("data");
}

// def AddLeNetModel(model, data):
void AddLeNetModel(ModelUtil &model, bool test) {
  // >>> conv1 = brew.conv(model, data, 'conv1', dim_in=1, dim_out=20, kernel=5)
  model.AddConvOps("data", "conv1", 1, 20, 1, 0, 5, test);

  // >>> pool1 = brew.max_pool(model, conv1, 'pool1', kernel=2, stride=2)
  model.predict.AddMaxPoolOp("conv1", "pool1", 2, 0, 2);

  // >>> conv2 = brew.conv(model, pool1, 'conv2', dim_in=20, dim_out=50,
  // kernel=5)
  model.AddConvOps("pool1", "conv2", 20, 50, 1, 0, 5, test);

  // >>> pool2 = brew.max_pool(model, conv2, 'pool2', kernel=2, stride=2)
  model.predict.AddMaxPoolOp("conv2", "pool2", 2, 0, 2);

  // >>> fc3 = brew.fc(model, pool2, 'fc3', dim_in=50 * 4 * 4, dim_out=500)
  model.AddFcOps("pool2", "fc3", 800, 500, test);

  // >>> fc3 = brew.relu(model, fc3, fc3)
  model.predict.AddReluOp("fc3", "fc3");

  // >>> pred = brew.fc(model, fc3, 'pred', 500, 10)
  model.AddFcOps("fc3", "pred", 500, 10, test);

  // >>> softmax = brew.softmax(model, pred, 'softmax')
  model.predict.AddSoftmaxOp("pred", "softmax");
}

// def AddAccuracy(model, softmax, label):
void AddAccuracy(ModelUtil &model) {
  // >>> accuracy = model.Accuracy([softmax, label], "accuracy")
  model.predict.AddAccuracyOp("softmax", "label", "accuracy");

  if (FLAGS_display) {
    model.predict.AddTimePlotOp("accuracy");
  }

  // >>> ITER = model.Iter("iter")
  model.AddIterOps();
}

// >>> def AddTrainingOperators(model, softmax, label):
void AddTrainingOperators(ModelUtil &model) {
  // >>> xent = model.LabelCrossEntropy([softmax, label], 'xent')
  model.predict.AddLabelCrossEntropyOp("softmax", "label", "xent");

  // >>> loss = model.AveragedLoss(xent, "loss")
  model.predict.AddAveragedLossOp("xent", "loss");

  if (FLAGS_display) {
    model.predict.AddShowWorstOp("softmax", "label", "data", 256, 0);
    model.predict.AddTimePlotOp("loss");
  }

  // >>> AddAccuracy(model, softmax, label)
  AddAccuracy(model);

  // >>> model.AddGradientOperators([loss])
  model.AddGradientOps();

  // >>> LR = model.LearningRate(ITER, "LR", base_lr=-0.1, policy="step",
  // stepsize=1, gamma=0.999 )
  model.predict.AddLearningRateOp("iter", "LR", 0.1);

  // >>> ONE = model.param_init_net.ConstantFill([], "ONE", shape=[1],
  // value=1.0)
  model.init.AddConstantFillOp({1}, 1.f, "ONE");
  model.predict.AddInput("ONE");

  // >>> for param in model.params:
  for (auto param : model.Params()) {
    // >>> param_grad = model.param_to_grad[param]
    // >>> model.WeightedSum([param, ONE, param_grad, LR], param)
    model.predict.AddWeightedSumOp({param, "ONE", param + "_grad", "LR"},
                                   param);
  }

  // Checkpoint causes problems on subsequent runs
  // >>> model.Checkpoint([ITER] + model.params, [],
  // std::vector<std::string> inputs({"iter"});
  // inputs.insert(inputs.end(), params.begin(), params.end());
  // model.predict.AddCheckpointOp(inputs, 20, "leveldb",
  //                         "mnist_lenet_checkpoint_%05d.leveldb");
}

// >>> def AddBookkeepingOperators(model):
void AddBookkeepingOperators(ModelUtil &model) {
  // >>> model.Print('accuracy', [], to_file=1)
  model.predict.AddPrintOp("accuracy", true);

  // >>> model.Print('loss', [], to_file=1)
  model.predict.AddPrintOp("loss", true);

  // >>> for param in model.params:
  for (auto param : model.Params()) {
    // >>> model.Summarize(param, [], to_file=1)
    model.predict.AddSummarizeOp(param, true);

    // >>> model.Summarize(model.param_to_grad[param], [], to_file=1)
    model.predict.AddSummarizeOp(param + "_grad", true);
  }
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

  std::cout << "train-db: " << FLAGS_train_db << std::endl;
  std::cout << "test-db: " << FLAGS_test_db << std::endl;
  std::cout << "train-runs: " << FLAGS_train_runs << std::endl;
  std::cout << "test-runs: " << FLAGS_test_runs << std::endl;
  std::cout << "force-cpu: " << (FLAGS_force_cpu ? "true" : "false")
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
    moveWindow("undercertain", 0, 0);
    resizeWindow("undercertain", 300, 300);
    setWindowTitle("undercertain", "uncertain but correct");
    moveWindow("overcertain", 0, 300);
    resizeWindow("overcertain", 300, 300);
    setWindowTitle("overcertain", "certain but incorrect");
    moveWindow("accuracy", 300, 0);
    resizeWindow("accuracy", 300, 300);
    moveWindow("loss", 300, 300);
    resizeWindow("loss", 300, 300);
  }

  // >>> from caffe2.python import core, cnn, net_drawer, workspace, visualize,
  // brew
  // >>> workspace.ResetWorkspace(root_folder)
  Workspace workspace("tmp");

  // >>> train_model = model_helper.ModelHelper(name="mnist_train",
  // arg_scope={"order": "NCHW"})
  NetDef train_init_model, train_predict_model;
  ModelUtil train(train_init_model, train_predict_model, "mnist_train");

  // >>> data, label = AddInput(train_model, batch_size=64,
  // db=os.path.join(data_folder, 'mnist-train-nchw-leveldb'),
  // db_type='leveldb')
  AddInput(train, 64, FLAGS_train_db, "leveldb");

  // >>> softmax = AddLeNetModel(train_model, data)
  AddLeNetModel(train, false);

  // >>> AddTrainingOperators(train_model, softmax, label)
  AddTrainingOperators(train);

  // >>> AddBookkeepingOperators(train_model)
  AddBookkeepingOperators(train);

  // >>> test_model = model_helper.ModelHelper(name="mnist_test",
  // arg_scope=arg_scope, init_params=False)
  NetDef test_init_model, test_predict_model;
  ModelUtil test(test_init_model, test_predict_model, "mnist_test");

  // >>> data, label = AddInput(test_model, batch_size=100,
  // db=os.path.join(data_folder, 'mnist-test-nchw-leveldb'), db_type='leveldb')
  AddInput(test, 100, FLAGS_test_db, "leveldb");

  // >>> softmax = AddLeNetModel(test_model, data)
  AddLeNetModel(test, true);

  // >>> AddAccuracy(test_model, softmax, label)
  AddAccuracy(test);

  // >>> deploy_model = model_helper.ModelHelper(name="mnist_deploy",
  // arg_scope=arg_scope, init_params=False)
  NetDef deploy_init_model, deploy_predict_model;
  ModelUtil deploy(deploy_init_model, deploy_predict_model, "mnist_model");
  deploy.predict.AddInput("data");

  // >>> AddLeNetModel(deploy_model, "data")
  AddLeNetModel(deploy, true);

#ifdef WITH_CUDA
  if (!FLAGS_force_cpu) {
    train.SetDeviceCUDA();
    test.SetDeviceCUDA();
  }
#endif

  std::cout << std::endl;

  // >>> workspace.RunNetOnce(train_model.param_init_net)
  auto initTrainNet = CreateNet(train.init.net, &workspace);
  initTrainNet->Run();

  // >>> workspace.CreateNet(train_model.net)
  auto predictTrainNet = CreateNet(train.predict.net, &workspace);

  std::cout << "training.." << std::endl;

  // >>> for i in range(total_iters):
  for (auto i = 1; i <= FLAGS_train_runs; i++) {
    // >>> workspace.RunNet(train_model.net.Proto().name)
    predictTrainNet->Run();

    // >>> accuracy[i] = workspace.FetchBlob('accuracy')
    // >>> loss[i] = workspace.FetchBlob('loss')
    if (i % 10 == 0) {
      auto accuracy =
          BlobUtil(*workspace.GetBlob("accuracy")).Get().data<float>()[0];
      auto loss = BlobUtil(*workspace.GetBlob("loss")).Get().data<float>()[0];
      std::cout << "step: " << i << " loss: " << loss
                << " accuracy: " << accuracy << std::endl;
    }
  }

  std::cout << std::endl;

  // >>> workspace.RunNetOnce(test_model.param_init_net)
  auto initTestNet = CreateNet(test.init.net, &workspace);
  initTestNet->Run();

  // >>> workspace.CreateNet(test_model.net)
  auto predictTestNet = CreateNet(test.predict.net, &workspace);

  std::cout << "testing.." << std::endl;

  // >>> for i in range(100):
  for (auto i = 1; i <= FLAGS_test_runs; i++) {
    // >>> workspace.RunNet(test_model.net.Proto().name)
    predictTestNet->Run();

    // >>> test_accuracy[i] = workspace.FetchBlob('accuracy')
    if (i % 10 == 0) {
      auto accuracy =
          BlobUtil(*workspace.GetBlob("accuracy")).Get().data<float>()[0];
      std::cout << "step: " << i << " accuracy: " << accuracy << std::endl;
    }
  }

  // with open(os.path.join(root_folder, "deploy_net.pbtxt"), 'w') as fid:
  // fid.write(str(deploy_model.net.Proto()))
  for (auto &param : deploy.predict.net.external_input()) {
    auto tensor = BlobUtil(*workspace.GetBlob(param)).Get();
    auto op = deploy.init.net.add_op();
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
    op->add_output(param);
  }
  deploy.predict.WriteText("tmp/mnist_predict_net.pbtxt");
  deploy.Write("tmp/mnist");
}

void predict_example() {
  std::vector<float> data_for_2(
      {0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
       0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
       0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
       0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
       0,   0,   0,   0,   0,   0,   0,   0,   0,   0.2, 0.6, 0.8, 0.9, 0.7,
       0.2, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
       0,   0,   0,   0,   0,   0,   0,   0,   0,   0.8, 0.3, 0.2, 0.2, 0.7,
       0.9, 0.4, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
       0,   0,   0,   0,   0,   0,   0,   0,   0,   0.3, 0,   0,   0,   0,
       0.4, 0.9, 0.3, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
       0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
       0,   0.4, 0.8, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
       0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
       0,   0,   0.8, 0.6, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
       0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
       0,   0,   0.2, 0.8, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
       0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
       0,   0,   0.1, 0.9, 0.3, 0,   0,   0,   0,   0,   0,   0,   0,   0,
       0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
       0,   0,   0,   0.7, 0.6, 0,   0,   0,   0,   0,   0,   0,   0,   0,
       0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
       0,   0,   0,   0.8, 0.6, 0,   0,   0,   0,   0,   0,   0,   0,   0,
       0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
       0,   0,   0,   0.9, 0.6, 0,   0,   0,   0,   0,   0,   0,   0,   0,
       0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
       0,   0,   0.2, 0.9, 0.3, 0,   0,   0,   0,   0,   0,   0,   0,   0,
       0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
       0,   0,   0.2, 0.9, 0.1, 0,   0,   0,   0,   0,   0,   0,   0,   0,
       0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0.2, 0.3, 0.3, 0,
       0,   0,   0.6, 0.7, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
       0,   0,   0,   0,   0,   0,   0,   0,   0,   0.6, 0.9, 0.9, 0.9, 0.9,
       0.4, 0.2, 0.9, 0.3, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
       0,   0,   0,   0,   0,   0,   0,   0,   0.7, 0.8, 0.1, 0,   0,   0.4,
       0.9, 0.9, 0.8, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
       0,   0,   0,   0,   0,   0,   0,   0.1, 0.9, 0.4, 0,   0,   0,   0,
       0.3, 0.9, 0.8, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
       0,   0,   0,   0,   0,   0,   0,   0.2, 0.9, 0.1, 0,   0,   0,   0.3,
       0.9, 0.8, 0.8, 0.7, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
       0,   0,   0,   0,   0,   0,   0,   0.1, 0.9, 0.1, 0,   0.2, 0.3, 0.9,
       0.8, 0.1, 0.1, 0.8, 0.7, 0.2, 0,   0,   0,   0,   0,   0,   0,   0,
       0,   0,   0,   0,   0,   0,   0,   0,   0.7, 0.9, 0.8, 0.9, 0.9, 0.6,
       0.1, 0,   0,   0.1, 0.5, 0.9, 0.7, 0.2, 0.1, 0,   0,   0,   0,   0,
       0,   0,   0,   0,   0,   0,   0,   0,   0,   0.5, 0.6, 0.3, 0,   0,
       0,   0,   0,   0,   0,   0.3, 0.8, 0.9, 0.2, 0,   0,   0,   0,   0,
       0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
       0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
       0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
       0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
       0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
       0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
       0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
       0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
       0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
       0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
       0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
       0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0});

  std::cout << "classifying image of decimal:";
  auto i = 0;
  for (auto d : data_for_2) {
    if (i % 28 == 0) std::cout << std::endl;
    std::cout << (d > 0 ? "[]" : "  ");
    i++;
  }
  std::cout << std::endl;

#ifdef WITH_CUDA
  DeviceOption option;
  option.set_device_type(CUDA);
  new CUDAContext(option);
#endif

  // setup perdictor
  NetDef init_model, predict_model;
  CAFFE_ENFORCE(ReadProtoFromFile("tmp/mnist_init_net.pb", &init_model));
  CAFFE_ENFORCE(ReadProtoFromFile("tmp/mnist_predict_net.pb", &predict_model));

#ifdef WITH_CUDA
  init_model.mutable_device_option()->set_device_type(CUDA);
  predict_model.mutable_device_option()->set_device_type(CUDA);
#endif

  // load parameters
  Workspace workspace("tmp");
  auto init_net = CreateNet(init_model, &workspace);
  init_net->Run();

// input image data for "2"
#ifdef WITH_CUDA
  auto data = workspace.CreateBlob("data")->GetMutable<TensorCUDA>();
#else
  auto data = workspace.CreateBlob("data")->GetMutable<TensorCPU>();
#endif
  TensorCPU input({1, 1, 28, 28}, data_for_2, NULL);
  data->ResizeLike(input);
  data->ShareData(input);

  // run predictor
  auto predict_net = CreateNet(predict_model, &workspace);
  predict_net->Run();

// read prediction
#ifdef WITH_CUDA
  auto softmax = TensorCPU(workspace.GetBlob("softmax")->Get<TensorCUDA>());
#else
  auto softmax = workspace.GetBlob("softmax")->Get<TensorCPU>();
#endif
  std::vector<float> probs(softmax.data<float>(),
                           softmax.data<float>() + softmax.size());
  auto max = std::max_element(probs.begin(), probs.end());
  auto label = std::distance(probs.begin(), max);
  std::cout << "predicted label: '" << label << "' with probability: " << *max
            << std::endl;
}

}  // namespace caffe2

int main(int argc, char **argv) {
  caffe2::GlobalInit(&argc, &argv);
  caffe2::run();
  // caffe2::predict_example();
  google::protobuf::ShutdownProtobufLibrary();
  return 0;
}
