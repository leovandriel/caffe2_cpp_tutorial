#ifndef TRAIN_H
#define TRAIN_H

#include <caffe2/core/db.h>
#include <caffe2/core/init.h>
#include <caffe2/core/net.h>

#include "caffe2/util/blob.h"
#include "caffe2/util/model.h"
#include "caffe2/util/net.h"
#include "caffe2/util/tensor.h"

namespace caffe2 {

enum { kRunTrain = 0, kRunValidate = 1, kRunTest = 2, kRunNum = 3 };

static std::map<int, std::string> name_for_run({
    {kRunTrain, "train"}, {kRunValidate, "validate"}, {kRunTest, "test"},
});

void run_trainer(int epochs, ModelUtil &train, ModelUtil &validate,
                 Workspace &workspace, clock_t &train_time,
                 clock_t &validate_time) {
  CreateNet(train.init.net, &workspace)->Run();
  CreateNet(validate.init.net, &workspace)->Run();

  auto train_net = CreateNet(train.predict.net, &workspace);
  auto validate_net = CreateNet(validate.predict.net, &workspace);

  auto last_time = clock();
  auto last_i = 0;
  auto sum_accuracy = 0.f, sum_loss = 0.f;

  for (auto i = 1; i <= epochs; i++) {
    train_time -= clock();
    train_net->Run();
    train_time += clock();

    sum_accuracy +=
        BlobUtil(*workspace.GetBlob("accuracy")).Get().data<float>()[0];
    sum_loss += BlobUtil(*workspace.GetBlob("loss")).Get().data<float>()[0];

    auto steps_time = (float)(clock() - last_time) / CLOCKS_PER_SEC;
    if (steps_time > 5 || i >= epochs) {
      auto iter = BlobUtil(*workspace.GetBlob("iter")).Get().data<int64_t>()[0];
      auto lr = BlobUtil(*workspace.GetBlob("lr")).Get().data<float>()[0];
      auto train_loss = sum_loss / (i - last_i),
           train_accuracy = sum_accuracy / (i - last_i);
      sum_loss = 0;
      sum_accuracy = 0;
      validate_time -= clock();
      validate_net->Run();
      validate_time += clock();
      auto validate_accuracy =
          BlobUtil(*workspace.GetBlob("accuracy")).Get().data<float>()[0];
      std::cout << "step: " << iter << "  rate: " << lr
                << "  loss: " << train_loss << "  accuracy: " << train_accuracy
                << " | " << validate_accuracy
                << "  step_time: " << std::setprecision(3)
                << steps_time / (i - last_i) << "s" << std::endl;
      last_i = i;
      last_time = clock();
    }
  }
}

void run_tester(int epochs, ModelUtil &test, Workspace &workspace,
                clock_t &test_time) {
  CreateNet(test.init.net, &workspace)->Run();
  auto test_net = CreateNet(test.predict.net, &workspace);

  auto sum_accuracy = 0.f, sum_loss = 0.f;
  auto test_step = 10;
  for (auto i = 1; i <= epochs; i++) {
    test_time -= clock();
    test_net->Run();
    test_time += clock();

    sum_accuracy +=
        BlobUtil(*workspace.GetBlob("accuracy")).Get().data<float>()[0];
    sum_loss += BlobUtil(*workspace.GetBlob("loss")).Get().data<float>()[0];

    if (i % test_step == 0) {
      auto loss = sum_loss / test_step, accuracy = sum_accuracy / test_step;
      sum_loss = 0;
      sum_accuracy = 0;
      std::cout << "step: " << i << " loss: " << loss
                << " accuracy: " << accuracy << std::endl;
    }
  }
}

}  // namespace caffe2

#endif  // TRAIN_H
