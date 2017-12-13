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

  void AddDatabaseOps(const std::string &name, const std::string &data,
                      const std::string &db, const std::string &db_type,
                      int batch_size);
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
	//operators with initializable parameters
  void AddFcOps(const std::string &input, const std::string &output,
                int in_size, int out_size, bool test = false);
  void AddConvOps(const std::string &input, const std::string &output,
                  int in_size, int out_size, int stride, int padding,
                  int kernel, bool test = false);
  void AddSpatialBNOps(const std::string & input, const std::string &output,
				int size, float epsilon = 1e-5f, float momentum = 0.9, bool test = false);
  //operators without initializable parameters
  void AddConstantFillOp(const std::vector<int>& shape,
                                 const std::string& param);
  void AddXavierFillOp(const std::vector<int>& shape,
                               const std::string& param);
  void AddUniformFillOp(const std::vector<int>& shape, float min,
                                float max, const std::string& param);
  void AddGaussianFillOp(const std::vector<int>& shape, float mean,
										float std, const std::string& param);
  void AddConstantFillOp(const std::vector<int>& shape, float value,
                                 const std::string& param);
  void AddConstantFillOp(const std::vector<int>& shape, int64_t value,
                                 const std::string& param);
  void AddConstantFillWithOp(float value, const std::string& input,
                                     const std::string& output);
  void AddVectorFillOp(const std::vector<int>& values,
                               const std::string& name);
  void AddGivenTensorFillOp(const TensorCPU& tensor,
                                    const std::string& name);
  void AddReluOp(const std::string& input, const std::string& output);
  void AddLeakyReluOp(const std::string& input,
								const std::string& output, float alpha);
  void AddReshapeOp(const std::string& input, const std::string& output,
                            const std::vector<int>& shape);
  void AddSigmoidOp(const std::string& input,
								const std::string& output);
  void AddLrnOp(const std::string& input, const std::string& output,
                        int size, float alpha, float beta, float bias,
                        const std::string& order = "NCHW");
  void AddMaxPoolOp(const std::string& input, const std::string& output,
                            int stride, int padding, int kernel,
                            const std::string& order = "NCHW");
  void AddAveragePoolOp(const std::string& input,
                                const std::string& output, int stride,
                                int padding, int kernel,
                                const std::string& order = "NCHW");
  void AddDropoutOp(const std::string& input, const std::string& output,
                            float ratio, bool test = false);
  void AddSoftmaxOp(const std::string& input, const std::string& output,
                            int axis = 1);
  void AddConcatOp(const std::vector<std::string>& inputs,
                           const std::string& output,
                           const std::string& order = "NCHW");
  void AddMulOp(const std::vector<std::string>& inputs,
                        const std::string& output, int broadcast = 1,
                        int axis = 1);
  void AddAddOp(const std::vector<std::string>& inputs,
                        const std::string& output, int broadcast = 1,
                        int axis = 1);
  void AddScaleOp(const std::string& input, const std::string& output,
                          float scale);
  void AddWeightedSumOp(const std::vector<std::string>& inputs,
                                const std::string& sum);
  void AddCopyOp(const std::string& input, const std::string& output);
  void AddSquaredL2DistanceOp(const std::vector<std::string> & inputs,
						const std::string& output);
  void AddPowOp(const std::string & input, const std::string & output, float exponent);
  void AddSubOp(const std::vector<std::string> & inputs, const std::string & output, int broadcast = 1, int axis = 1);
  void AddSoftmaxWithLossOp(const std::vector<std::string>& inputs,
								const std::vector<std::string>& outputs);
  void AddAveragedLossOp(const std::string& input,
                                 const std::string& loss);
  void AddLearningRateOp(const std::string& iter,
                                 const std::string& rate, float base_rate,
                                 float gamma = 0.999f,  int stepsize = 1.0);
  //TODO:
  void AddStopGradientOp(const std::string& param);

  std::vector<std::string> Params() { return predict.CollectParams(); }

  void Split(const std::string &layer, ModelUtil &firstModel,
             ModelUtil &secondModel, bool force_cpu, bool inclusive = true);
  void CopyTrain(const std::string &layer, int out_size,
                 ModelUtil &train) const;
  void CopyTest(ModelUtil &test) const;
  void CopyDeploy(ModelUtil &deploy, Workspace &workspace) const;

  size_t Write(const std::string &path_prefix) const;
  size_t Read(const std::string &path_prefix);
  void SetName(const std::string &name);
  void SetDeviceCUDA();
  std::string Short();

 public:
  NetUtil init;
  NetUtil predict;
};

}  // namespace caffe2

#endif  // UTIL_MODEL_H
