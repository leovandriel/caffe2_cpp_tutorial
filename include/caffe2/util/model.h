#ifndef UTIL_MODEL_H
#define UTIL_MODEL_H

#include <caffe2/core/net.h>
#include "caffe2/util/net.h"

namespace caffe2 {

class ModelUtil {
 public:
  ModelUtil(NetDef &init_net, NetDef &predict_net, const std::string &name = "")
      : init(init_net), predict(predict_net) {
    if (name.size()) {
      SetName(name);
    }
  }
  ModelUtil(NetUtil &init, NetUtil &predict) : init(init), predict(predict) {}

  void SetName(const std::string &name);
  void AddDatabaseOps(const std::string &name, const std::string &data,
                      const std::string &db, const std::string &db_type,
                      int batch_size);
  void AddGradientOps();
  void AddXentOps(const std::string &output);
  void AddIterOps();

  void AddSgdOps();
  void AddMomentumOps();
  void AddAdagradOps();
  void AddAdamOps();
  void AddRmsPropOps();
  void AddOptimizerOps(std::string &optimizer);

  void AddTestOps(const std::string &output);
  void AddTrainOps(const std::string &output, float base_rate,
                   std::string &optimizer);

  void AddFcOps(const std::string &input, const std::string &output,
                int in_size, int out_size, bool test = false);
  void AddConvOps(const std::string &input, const std::string &output,
                  int in_size, int out_size, int stride, int padding,
                  int kernel, bool test = false);

  std::vector<std::string> Params() { return predict.CollectParams(); }

 public:
  NetUtil init;
  NetUtil predict;
};

}  // namespace caffe2

#endif  // UTIL_MODEL_H
