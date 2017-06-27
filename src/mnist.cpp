#include "caffe2/core/init.h"
#include "caffe2/core/operator_gradient.h"

#include "util/print.h"

CAFFE2_DEFINE_string(train_db, "res/mnist-train-nchw-leveldb", "The given path to the training leveldb.");
CAFFE2_DEFINE_string(test_db, "res/mnist-test-nchw-leveldb", "The given path to the testing leveldb.");
CAFFE2_DEFINE_int(train_runs, 200, "The of training runs.");
CAFFE2_DEFINE_int(test_runs, 100, "The of test runs.");

namespace caffe2 {

// >> def AddInput(model, batch_size, db, db_type):
void AddInput(NetDef &initModel, NetDef &predictModel, int batch_size, const std::string &db, const std::string& db_type) {
  // Setup database connection
  {
    auto op = initModel.add_op();
    op->set_type("CreateDB");
    auto arg1 = op->add_arg();
    arg1->set_name("db_type");
    arg1->set_s(db_type);
    auto arg2 = op->add_arg();
    arg2->set_name("db");
    arg2->set_s(db);
    op->add_output("dbreader");
  }

  // >>> data_uint8, label = model.TensorProtosDBInput([], ["data_uint8", "label"], batch_size=batch_size, db=db, db_type=db_type)
  {
    auto op = predictModel.add_op();
    op->set_type("TensorProtosDBInput");
    auto arg = op->add_arg();
    arg->set_name("batch_size");
    arg->set_i(batch_size);
    op->add_input("dbreader");
    op->add_output("data_uint8");
    op->add_output("label");
  }

  // >>> data = model.Cast(data_uint8, "data", to=core.DataType.FLOAT)
  {
    auto op = predictModel.add_op();
    op->set_type("Cast");
    auto arg = op->add_arg();
    arg->set_name("to");
    arg->set_i(TensorProto_DataType_FLOAT);
    op->add_input("data_uint8");
    op->add_output("data");
  }

  // >>> data = model.Scale(data, data, scale=float(1./256))
  {
    auto op = predictModel.add_op();
    op->set_type("Scale");
    auto arg = op->add_arg();
    arg->set_name("scale");
    arg->set_f(static_cast<float>(1) / static_cast<float>(256));
    op->add_input("data");
    op->add_output("data");
  }

  // >>> data = model.StopGradient(data, data)
  {
    auto op = predictModel.add_op();
    op->set_type("StopGradient");
    op->add_input("data");
    op->add_output("data");
  }
}

// def AddLeNetModel(model, data):
void AddLeNetModel(NetDef &initModel, NetDef &predictModel, std::vector<OperatorDef *> &gradient_ops, bool training) {
  // >>> conv1 = brew.conv(model, data, 'conv1', dim_in=1, dim_out=20, kernel=5)
  {
    auto op = predictModel.add_op();
    op->set_type("Conv");
    auto arg = op->add_arg();
    arg->set_name("kernel");
    arg->set_i(5);
    op->add_input("data");
    op->add_input("conv1_w");
    op->add_input("conv1_b");
    op->add_output("conv1");
    gradient_ops.push_back(op);
  }
  if (training) {
    auto op = initModel.add_op();
    op->set_type("XavierFill");
    auto arg = op->add_arg();
    arg->set_name("shape");
    arg->add_ints(20);
    arg->add_ints(1);
    arg->add_ints(5);
    arg->add_ints(5);
    op->add_output("conv1_w");
  } else {
    initModel.add_external_input("conv1_w");
  }
  if (training) {
    auto op = initModel.add_op();
    op->set_type("ConstantFill");
    auto arg = op->add_arg();
    arg->set_name("shape");
    arg->add_ints(20);
    op->add_output("conv1_b");
  } else {
    initModel.add_external_input("conv1_b");
  }

  // >>> pool1 = brew.max_pool(model, conv1, 'pool1', kernel=2, stride=2)
  {
    auto op = predictModel.add_op();
    op->set_type("MaxPool");
    auto arg1 = op->add_arg();
    arg1->set_name("kernel");
    arg1->set_i(2);
    auto arg2 = op->add_arg();
    arg2->set_name("stride");
    arg2->set_i(2);
    op->add_input("conv1");
    op->add_output("pool1");
    gradient_ops.push_back(op);
  }

  // >>> conv2 = brew.conv(model, pool1, 'conv2', dim_in=20, dim_out=50, kernel=5)
  {
    auto op = predictModel.add_op();
    op->set_type("Conv");
    auto arg = op->add_arg();
    arg->set_name("kernel");
    arg->set_i(5);
    op->add_input("pool1");
    op->add_input("conv2_w");
    op->add_input("conv2_b");
    op->add_output("conv2");
    gradient_ops.push_back(op);
  }
  if (training) {
    auto op = initModel.add_op();
    op->set_type("XavierFill");
    auto arg = op->add_arg();
    arg->set_name("shape");
    arg->add_ints(50);
    arg->add_ints(20);
    arg->add_ints(5);
    arg->add_ints(5);
    op->add_output("conv2_w");
  } else {
    initModel.add_external_input("conv2_w");
  }
  if (training) {
    auto op = initModel.add_op();
    op->set_type("ConstantFill");
    auto arg = op->add_arg();
    arg->set_name("shape");
    arg->add_ints(50);
    op->add_output("conv2_b");
  } else {
    initModel.add_external_input("conv2_b");
  }

  // >>> pool2 = brew.max_pool(model, conv2, 'pool2', kernel=2, stride=2)
  {
    auto op = predictModel.add_op();
    op->set_type("MaxPool");
    auto arg1 = op->add_arg();
    arg1->set_name("kernel");
    arg1->set_i(2);
    auto arg2 = op->add_arg();
    arg2->set_name("stride");
    arg2->set_i(2);
    op->add_input("conv2");
    op->add_output("pool2");
    gradient_ops.push_back(op);
  }

  // >>> fc3 = brew.fc(model, pool2, 'fc3', dim_in=50 * 4 * 4, dim_out=500)
  {
    auto op = predictModel.add_op();
    op->set_type("FC");
    op->add_input("pool2");
    op->add_input("fc3_w");
    op->add_input("fc3_b");
    op->add_output("fc3");
    gradient_ops.push_back(op);
  }
  if (training) {
    auto op = initModel.add_op();
    op->set_type("XavierFill");
    auto arg = op->add_arg();
    arg->set_name("shape");
    arg->add_ints(500);
    arg->add_ints(800);
    op->add_output("fc3_w");
  } else {
    initModel.add_external_input("fc3_w");
  }
  if (training) {
    auto op = initModel.add_op();
    op->set_type("ConstantFill");
    auto arg = op->add_arg();
    arg->set_name("shape");
    arg->add_ints(500);
    op->add_output("fc3_b");
  } else {
    initModel.add_external_input("fc3_b");
  }

  // >>> fc3 = brew.relu(model, fc3, fc3)
  {
    auto op = predictModel.add_op();
    op->set_type("Relu");
    op->add_input("fc3");
    op->add_output("fc3");
    gradient_ops.push_back(op);
  }

  // >>> pred = brew.fc(model, fc3, 'pred', 500, 10)
  {
    auto op = predictModel.add_op();
    op->set_type("FC");
    op->add_input("fc3");
    op->add_input("pred_w");
    op->add_input("pred_b");
    op->add_output("pred");
    gradient_ops.push_back(op);
  }
  if (training) {
    auto op = initModel.add_op();
    op->set_type("XavierFill");
    auto arg = op->add_arg();
    arg->set_name("shape");
    arg->add_ints(10);
    arg->add_ints(500);
    op->add_output("pred_w");
  } else {
    initModel.add_external_input("pred_w");
  }
  if (training) {
    auto op = initModel.add_op();
    op->set_type("ConstantFill");
    auto arg = op->add_arg();
    arg->set_name("shape");
    arg->add_ints(10);
    op->add_output("pred_b");
  } else {
    initModel.add_external_input("pred_b");
  }

  // >>> softmax = brew.softmax(model, pred, 'softmax')
  {
    auto op = predictModel.add_op();
    op->set_type("Softmax");
    op->add_input("pred");
    op->add_output("softmax");
    gradient_ops.push_back(op);
  }
}

// def AddAccuracy(model, softmax, label):
void AddAccuracy(NetDef &initModel, NetDef &predictModel) {
  // >>> accuracy = model.Accuracy([softmax, label], "accuracy")
  {
    auto op = predictModel.add_op();
    op->set_type("Accuracy");
    op->add_input("softmax");
    op->add_input("label");
    op->add_output("accuracy");
  }

  // Moved ITER to AddAccuracy function, so it's also available on test runs
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
    arg3->set_i(TensorProto_DataType_INT64);
    op->add_output("ITER");
  }
  // >>> ITER = model.Iter("iter")
  {
    auto op = predictModel.add_op();
    op->set_type("Iter");
    op->add_input("ITER");
    op->add_output("ITER");
  }
}

// >>> def AddTrainingOperators(model, softmax, label):
void AddTrainingOperators(NetDef &initModel, NetDef &predictModel, std::vector<string> params, std::vector<OperatorDef *> &gradient_ops) {
  // >>> xent = model.LabelCrossEntropy([softmax, label], 'xent')
  {
    auto op = predictModel.add_op();
    op->set_type("LabelCrossEntropy");
    op->add_input("softmax");
    op->add_input("label");
    op->add_output("xent");
    gradient_ops.push_back(op);
  }

  // >>> loss = model.AveragedLoss(xent, "loss")
  {
    auto op = predictModel.add_op();
    op->set_type("AveragedLoss");
    op->add_input("xent");
    op->add_output("loss");
    gradient_ops.push_back(op);
  }

  // >>> AddAccuracy(model, softmax, label)
  AddAccuracy(initModel, predictModel);

  // >>> model.AddGradientOperators([loss])
  {
    auto op = predictModel.add_op();
    op->set_type("ConstantFill");
    auto arg = op->add_arg();
    arg->set_name("value");
    arg->set_f(1.0);
    op->add_input("loss");
    op->add_output("loss_grad");
    op->set_is_gradient_op(true);
  }
  std::reverse(gradient_ops.begin(), gradient_ops.end());
  for (auto op: gradient_ops) {
    vector<GradientWrapper> output(op->output_size());
    for (auto i = 0; i < output.size(); i++) {
      output[i].dense_ = op->output(i) + "_grad";
    }
    GradientOpsMeta meta = GetGradientForOp(*op, output);
    auto grad = predictModel.add_op();
    grad->CopyFrom(meta.ops_[0]);
    grad->set_is_gradient_op(true);
  }

  // >>> LR = model.LearningRate(ITER, "LR", base_lr=-0.1, policy="step", stepsize=1, gamma=0.999 )
  {
    auto op = predictModel.add_op();
    op->set_type("LearningRate");
    auto arg1 = op->add_arg();
    arg1->set_name("policy");
    arg1->set_s("step");
    auto arg2 = op->add_arg();
    arg2->set_name("stepsize");
    arg2->set_i(1);
    auto arg3 = op->add_arg();
    arg3->set_name("base_lr");
    arg3->set_f(-0.1);
    auto arg4 = op->add_arg();
    arg4->set_name("gamma");
    arg4->set_f(0.999);
    op->add_input("ITER");
    op->add_output("LR");
  }

  // >>> ONE = model.param_init_net.ConstantFill([], "ONE", shape=[1], value=1.0)
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

  // >>> for param in model.params:
  for (auto param: params) {
    // >>> param_grad = model.param_to_grad[param]
    // >>> model.WeightedSum([param, ONE, param_grad, LR], param)
    {
      auto op = predictModel.add_op();
      op->set_type("WeightedSum");
      op->add_input(param);
      op->add_input("ONE");
      op->add_input(param + "_grad");
      op->add_input("LR");
      op->add_output(param);
    }
  }

  return; // Checkpoint causes problems on subsequent runs

  // >>> model.Checkpoint([ITER] + model.params, [],
  {
    auto op = predictModel.add_op();
    op->set_type("Checkpoint");
    auto arg1 = op->add_arg();
    arg1->set_name("every");
    arg1->set_i(20);
    auto arg2 = op->add_arg();
    arg2->set_name("db_type");
    arg2->set_s("leveldb");
    auto arg3 = op->add_arg();
    arg3->set_name("db");
    arg3->set_s("mnist_lenet_checkpoint_%05d.leveldb");
    op->add_input("ITER");
    for (auto param: params) {
      op->add_input(param);
    }
  }
}

// >>> def AddBookkeepingOperators(model):
void AddBookkeepingOperators(NetDef &initModel, NetDef &predictModel, std::vector<string> params) {
  // >>> model.Print('accuracy', [], to_file=1)
  {
    auto op = predictModel.add_op();
    op->set_type("Print");
    auto arg = op->add_arg();
    arg->set_name("to_file");
    arg->set_i(1);
    op->add_input("accuracy");
  }

  // >>> model.Print('loss', [], to_file=1)
  {
    auto op = predictModel.add_op();
    op->set_type("Print");
    auto arg = op->add_arg();
    arg->set_name("to_file");
    arg->set_i(1);
    op->add_input("loss");
  }

  // >>> for param in model.params:
  for (auto param: params) {
    // >>> model.Summarize(param, [], to_file=1)
    {
      auto op = predictModel.add_op();
      op->set_type("Summarize");
      auto arg = op->add_arg();
      arg->set_name("to_file");
      arg->set_i(1);
      op->add_input(param);
    }

    // >>> model.Summarize(model.param_to_grad[param], [], to_file=1)
    {
      auto op = predictModel.add_op();
      op->set_type("Summarize");
      auto arg = op->add_arg();
      arg->set_name("to_file");
      arg->set_i(1);
      op->add_input(param + "_grad");
    }
  }
}

void run() {
  std::cout << std::endl;
  std::cout << "## Caffe2 MNIST Tutorial ##" << std::endl;
  std::cout << "https://caffe2.ai/docs/tutorial-MNIST.html" << std::endl;
  std::cout << std::endl;

  std::cout << "train_db: " << FLAGS_train_db << std::endl;
  std::cout << "test_db: " << FLAGS_test_db << std::endl;
  std::cout << "train_runs: " << FLAGS_train_runs << std::endl;
  std::cout << "test_runs: " << FLAGS_test_runs << std::endl;

  // >>> from caffe2.python import core, cnn, net_drawer, workspace, visualize, brew
  // >>> workspace.ResetWorkspace(root_folder)
  Workspace workspace("tmp");

  // >>> train_model = model_helper.ModelHelper(name="mnist_train", arg_scope={"order": "NCHW"})
  NetDef initTrainModel;
  initTrainModel.set_name("mnist_train_init");
  NetDef predictTrainModel;
  predictTrainModel.set_name("mnist_train_predict");

  std::vector<string> params({"conv1_w", "conv1_b", "conv2_w", "conv2_b", "fc3_w", "fc3_b", "pred_w", "pred_b"});

  // >>> data, label = AddInput(train_model, batch_size=64, db=os.path.join(data_folder, 'mnist-train-nchw-leveldb'), db_type='leveldb')
  AddInput(initTrainModel, predictTrainModel, 64, FLAGS_train_db, "leveldb");

  // >>> softmax = AddLeNetModel(train_model, data)
  std::vector<OperatorDef *> gradient_ops;
  AddLeNetModel(initTrainModel, predictTrainModel, gradient_ops, true);

  // >>> AddTrainingOperators(train_model, softmax, label)
  AddTrainingOperators(initTrainModel, predictTrainModel, params, gradient_ops);

  // >>> AddBookkeepingOperators(train_model)
  AddBookkeepingOperators(initTrainModel, predictTrainModel, params);

  // print(initTrainModel);
  // print(predictTrainModel);

  // >>> test_model = model_helper.ModelHelper(name="mnist_test", arg_scope=arg_scope, init_params=False)
  NetDef initTestModel;
  initTestModel.set_name("mnist_test_init");
  NetDef predictTestModel;
  predictTestModel.set_name("mnist_test_predict");

  // >>> data, label = AddInput(test_model, batch_size=100, db=os.path.join(data_folder, 'mnist-test-nchw-leveldb'), db_type='leveldb')
  AddInput(initTestModel, predictTestModel, 100, FLAGS_test_db, "leveldb");

  // >>> softmax = AddLeNetModel(test_model, data)
  std::vector<OperatorDef *> gradient_ops_test;
  AddLeNetModel(initTestModel, predictTestModel, gradient_ops_test, false);

  // >>> AddAccuracy(test_model, softmax, label)
  AddAccuracy(initTestModel, predictTestModel);

  // >>> deploy_model = model_helper.ModelHelper(name="mnist_deploy", arg_scope=arg_scope, init_params=False)
  NetDef initDeployModel;
  initDeployModel.set_name("mnist_model_init");
  NetDef predictDeployModel;
  predictDeployModel.set_name("mnist_model_predict");

  // >>> AddLeNetModel(deploy_model, "data")
  std::vector<OperatorDef *> gradient_ops_deploy;
  AddLeNetModel(initDeployModel, predictDeployModel, gradient_ops_deploy, false);

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
      auto accuracy = workspace.GetBlob("accuracy")->Get<TensorCPU>().data<float>()[0];
      auto loss = workspace.GetBlob("loss")->Get<TensorCPU>().data<float>()[0];
      std::cout << "step: " << i << " loss: " << loss << " accuracy: " << accuracy << std::endl;
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
      auto accuracy = workspace.GetBlob("accuracy")->Get<TensorCPU>().data<float>()[0];
      std::cout << "step: " << i << " accuracy: " << accuracy << std::endl;
    }
  }

  // with open(os.path.join(root_folder, "deploy_net.pbtxt"), 'w') as fid:
    // fid.write(str(deploy_model.net.Proto()))
  std::vector<string> external(initDeployModel.external_input().begin(), initDeployModel.external_input().end());
  for (auto &param: external) {
    auto &tensor = workspace.GetBlob(param)->Get<TensorCPU>();
    auto op = initDeployModel.add_op();
    op->set_type("GivenTensorFill");
    auto arg1 = op->add_arg();
    arg1->set_name("shape");
    for (auto d: tensor.dims()) {
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
  // print(initDeployModel);
  // print(predictDeployModel);
}

}  // namespace caffe2

int main(int argc, char** argv) {
  caffe2::GlobalInit(&argc, &argv);
  caffe2::run();
  google::protobuf::ShutdownProtobufLibrary();
  return 0;
}
