#ifndef TRAIN_H
#define TRAIN_H

#include <caffe2/core/db.h>
#include <caffe2/core/init.h>
#include <caffe2/core/net.h>

#include "caffe2/util/blob.h"
#include "caffe2/util/model.h"
#include "caffe2/util/net.h"
#include "caffe2/util/progress.h"
#include "caffe2/util/tensor.h"

namespace caffe2 {

enum { kRunTrain = 0, kRunValidate = 1, kRunTest = 2, kRunNum = 3 };

static std::map<int, std::string> name_for_run({
    {kRunTrain, "train"},
    {kRunValidate, "validate"},
    {kRunTest, "test"},
});

void run_trainer(int iters, ModelUtil &train, ModelUtil &validate,
                 Workspace &workspace, clock_t &train_time,
                 clock_t &validate_time, bool verbose = true) {
  CAFFE_ENFORCE(workspace.RunNetOnce(train.init.net));
  CAFFE_ENFORCE(workspace.RunNetOnce(validate.init.net));

  CAFFE_ENFORCE(workspace.CreateNet(train.predict.net));
  CAFFE_ENFORCE(workspace.CreateNet(validate.predict.net));

  auto last_time = clock();
  auto last_i = 0;
  auto sum_accuracy = 0.f, sum_loss = 0.f;
  Progress progress(iters);

  for (auto i = 1; i <= iters; i++, progress.update()) {
    train_time -= clock();
    CAFFE_ENFORCE(workspace.RunNet(train.predict.net.name()));
    train_time += clock();

    sum_accuracy +=
        BlobUtil(*workspace.GetBlob("accuracy")).Get().data<float>()[0];
    sum_loss += BlobUtil(*workspace.GetBlob("loss")).Get().data<float>()[0];

    auto steps_time = (float)(clock() - last_time) / CLOCKS_PER_SEC;
    if (steps_time > 5 || i >= iters) {
      auto iter = BlobUtil(*workspace.GetBlob("iter")).Get().data<int64_t>()[0];
      auto lr = BlobUtil(*workspace.GetBlob("lr")).Get().data<float>()[0];
      auto train_loss = sum_loss / (i - last_i),
           train_accuracy = sum_accuracy / (i - last_i);
      sum_loss = 0;
      sum_accuracy = 0;
      validate_time -= clock();
      CAFFE_ENFORCE(workspace.RunNet(validate.predict.net.name()));
      validate_time += clock();
      auto validate_accuracy =
          BlobUtil(*workspace.GetBlob("accuracy")).Get().data<float>()[0];
      if (verbose)
        std::cout << "step: " << iter << "  rate: " << lr
                  << "  loss: " << train_loss
                  << "  accuracy: " << train_accuracy << " | "
                  << validate_accuracy
                  << "  step_time: " << std::setprecision(3)
                  << steps_time / (i - last_i) << "s" << std::endl;
      last_i = i;
      last_time = clock();
    }
  }
  progress.wipe();
}

void run_tester(int iters, ModelUtil &test, Workspace &workspace,
                clock_t &test_time, bool show_matrix = false,
                bool verbose = true) {
  CAFFE_ENFORCE(workspace.RunNetOnce(test.init.net));
  CAFFE_ENFORCE(workspace.CreateNet(test.predict.net));

  auto &output_name = test.predict.net.external_output(0);

  auto sum_accuracy = 0.f, sum_loss = 0.f;
  auto test_step = 10, batch_length = 0;
  std::map<std::pair<int, int>, int> counts;
  Progress progress(iters);

  for (auto i = 1; i <= iters; i++, progress.update()) {
    test_time -= clock();
    CAFFE_ENFORCE(workspace.RunNet(test.predict.net.name()));
    test_time += clock();

    sum_accuracy +=
        BlobUtil(*workspace.GetBlob("accuracy")).Get().data<float>()[0];
    sum_loss += BlobUtil(*workspace.GetBlob("loss")).Get().data<float>()[0];

    auto label =
        caffe2::BlobUtil(*workspace.GetBlob("label")).Get().data<int>();
    auto output = caffe2::BlobUtil(*workspace.GetBlob(output_name)).Get();
    auto batch_count = output.dim(0);
    batch_length = output.size() / batch_count;
    auto data = output.data<float>();
    for (int i = 0; i < batch_count; i++, data += batch_length) {
      auto max =
          std::distance(data, std::max_element(data, data + batch_length));
      counts[{label[i], max}]++;
    }

    if (i % test_step == 0) {
      auto loss = sum_loss / test_step, accuracy = sum_accuracy / test_step;
      sum_loss = 0;
      sum_accuracy = 0;
      if (verbose)
        std::cout << "step: " << i << " loss: " << loss
                  << " accuracy: " << accuracy << std::endl;
    }
  }
  progress.wipe();

  if (show_matrix) {
    std::cout << "  %  ";
    for (int j = 0; j < batch_length; j++) {
      std::cout << std::setw(6) << j;
    }
    std::cout << std::endl;
    for (int i = 0; i < batch_length; i++) {
      auto sum = 0;
      for (int j = 0; j < batch_length; j++) {
        sum += counts[{i, j}];
      }
      if (sum > 0) {
        std::cout << std::setw(4) << i << ":";
        for (int j = 0; j < batch_length; j++) {
          std::cout << std::fixed << std::setw(6) << std::setprecision(1)
                    << (100.0 * counts[{i, j}] / sum);
        }
        std::cout << " (" << sum << ")" << std::endl;
      }
    }
  }
}

}  // namespace caffe2

#endif  // TRAIN_H
