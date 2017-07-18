#ifndef UTIL_MODEL_H
#define UTIL_MODEL_H

#include "caffe2/core/net.h"
#include "caffe2/util/net.h"

namespace caffe2 {

class ModelUtil {
 public:
  ModelUtil(NetDef& init_net, NetDef& predict_net):
    init_(init_net),
    predict_(predict_net) {}

  void AddDatabaseOps(const std::string &name, const std::string &data, const std::string &db, const std::string &db_type, int batch_size);
  void AddXentOps(const std::string &output);
  void AddIterLrOps(float base_rate);

  void AddSgdOps();
  void AddMomentumOps();
  void AddAdagradOps();
  void AddAdamOps();
  void AddOptimizerOps(std::string &optimizer);

  void AddTestOps(const std::string &output);
  void AddTrainOps(const std::string &output, float base_rate, std::string &optimizer);

 protected:
   NetUtil init_;
   NetUtil predict_;
};

}  // namespace caffe2

#endif  // UTIL_MODEL_H
