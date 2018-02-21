#ifndef UTIL_MODEL_H
#define UTIL_MODEL_H

#include <caffe2/core/net.h>
#include "caffe2/util/net.h"

class ModelMeta;

namespace caffe2 {

class ModelUtil {
 public:
  ModelUtil(NetDef &init_net, NetDef &predict_net,
            const std::string &name = "");
  ModelUtil(NetUtil &init, NetUtil &predict);
  ~ModelUtil();

  void AddDatabaseOps(const std::string &name, const std::string &data,
                      const std::string &db, const std::string &db_type,
                      int batch_size);
  void AddXentOps(const std::string &output);
  void AddIterOps();

  void AddSgdOps();
  void AddSgdOps(ModelUtil & model);
  void AddMomentumOps();
  void AddMomentumOps(ModelUtil & model);
  void AddAdagradOps();
  void AddAdagradOps(ModelUtil & model);
  void AddAdamOps();
  void AddAdamOps(ModelUtil & model);
  void AddRmsPropOps();
  void AddRmsPropOps(ModelUtil & model);
  void AddOptimizerOps(std::string &optimizer);
  void AddOptimizerOps(std::string &optimizer,ModelUtil & model);

  void AddTestOps(const std::string &output);
  void AddTrainOps(const std::string &output, float base_rate,
                   std::string &optimizer);
  void AddTensorProtosDbInputOp(const std::string& reader,
                                        const std::string& data,
                                        const std::string& label,
                                        int batch_size);
	//operators with initializable parameters
  void AddFcOps(const std::string &input, const std::string &output,
                int in_size, int out_size, bool test = false);
  void AddConvOps(const std::string &input, const std::string &output,
                  int in_size, int out_size, int stride, int padding,
                  int kernel, int group = 0, bool test = false);
  void AddConv3DOps(const std::string& input, const std::string& output, 
					int in_size, int out_size, 
					std::vector<int> strides, std::vector<int> pads, std::vector<int> kernels, 
					bool test = false);
  void AddConvTransposeOps(const std::string &input, const std::string &output,
                  int in_size, int out_size, int stride, int padding,
                  int kernel, bool test = false);
  void AddSpatialBNOps(const std::string & input, const std::string &output,
				int size, float epsilon = 1e-5f, float momentum = 0.9, bool test = false);
  void AddSumOp(const std::vector<std::string>& inputs,
                               const std::string& sum);
  void AddGradientOps(const std::string & loss);
  void AddGradientOps(const std::string & loss, ModelUtil & model);
  //operators without initializable parameters
  void AddConstantFillOp(const std::vector<int>& shape,
                                 const std::string& output);
  void AddXavierFillOp(const std::vector<int>& shape,
                               const std::string& param);
  void AddUniformFillOp(const std::vector<int>& shape, float min,
                                float max, const std::string& param);
  void AddGaussianFillOp(const std::vector<int>& shape, float mean,
										float std, const std::string& param);
  void AddConstantFloatFillOp(const std::vector<int>& shape, float value,
                                 const std::string& output);
  void AddConstantIntFillOp(const std::vector<int>& shape, int value,
								const std::string& output);
  void AddConstantLongFillOp(const std::vector<int>& shape, int64_t value,
                                 const std::string& output);
  void AddConstantFillWithOp(float value, const std::string& input,
                                     const std::string& output);
  void AddVectorFillOp(const std::vector<int>& values,
                               const std::string& name);
  void AddGivenTensorFillOp(const TensorCPU& tensor,
                                    const std::string& name);
  void AddTransposeOp(const std::string& input, const std::string& output,
							  const std::vector<int>& axes);
  void AddReluOp(const std::string& input, const std::string& output);
  void AddLeakyReluOp(const std::string& input,
								const std::string& output, float alpha);
  void AddSliceOp(
    const std::string& input, const std::string& output,
    const std::vector<std::pair<int, int>>& ranges);
  void AddReshapeOp(const std::string& input, const std::string& output,
                            const std::vector<int>& shape);
  void AddSigmoidOp(const std::string& input,
								const std::string& output);
  void AddTanhOp(const std::string& input,
                                const std::string& output);
  void AddLrnOp(const std::string& input, const std::string& output,
                        int size, float alpha, float beta, float bias,
                        const std::string& order = "NCHW");
  void AddMaxPoolOp(const std::string& input, const std::string& output,
                            int stride, int padding, int kernel,
                            const std::string& order = "NCHW");
  void AddMaxPoolWithIndexOp(const std::string& input, const std::string& output, const std::string& index,
                            std::vector<int> strides, std::vector<int> pads, std::vector<int> kernels,
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
  void AddConcatOp(const std::vector<std::string>& inputs,
                           const std::string& output, int axis);
  void AddMulOp(const std::vector<std::string>& inputs,
                        const std::string& output, int broadcast = 1,
                        int axis = 1);
  void AddAddOp(const std::vector<std::string>& inputs,
                        const std::string& output, int broadcast = 1,
                        int axis = 1);
  void AddModOp(const std::string& input,
						const std::string& output, int divisor);
  void AddScaleOp(const std::string& input, const std::string& output,
                          float scale);
  void AddWeightedSumOp(const std::vector<std::string>& inputs,
                                const std::string& sum);
  void AddCopyOp(const std::string& input, const std::string& output);
  void AddCopyGPUToCPUOp(const std::string& input,const std::string& output);
  void AddCopyCPUToGPUOp(const std::string& input,const std::string& output);
  void AddPrintOp(const std::string& param, bool to_file);
  void AddSquaredL2DistanceOp(const std::vector<std::string> & inputs,
						const std::string& output);
  void AddL1DistanceOp(const std::vector<std::string> & inputs,
                        const std::string& output);
  void AddReduceBackMeanOp(const std::string& input, const std::string& mean);
  void AddReduceBackSumOp(const std::string& input, const std::string& sum);
  void AddReduceTailSumOp(const std::string& input, const std::string& sum);
  void AddPowOp(const std::string & input, const std::string & output, float exponent);
  void AddSubOp(const std::vector<std::string> & inputs, const std::string & output, int broadcast = 1, int axis = 1);
  void AddUpsampleNearestOp(const std::string & input,const std::string & output, float scale = 2);
  void AddSoftmaxWithLossOp(const std::vector<std::string>& inputs,
								const std::vector<std::string>& outputs,int axis = 1);
  void AddAveragedLossOp(const std::string& input,
                                 const std::string& loss);
  void AddLearningRateOp(const std::string& iter,
                                 const std::string& rate, float base_rate,
                                 float gamma = 0.999f,  int stepsize = 1.0);
  //TODO:
  void AddCastOp(const std::string& input,
                                const std::string& output,
                                TensorProto::DataType type);
  void AddStopGradientOp(const std::string& param);
  void AddSaveOp(const std::vector<std::string>& inputs,const std::string& type,
						const std::string & path);
  void AddLoadOp(const std::vector<std::string>& outputs,const std::string& type,
						const std::string & path);

  std::vector<std::string> Params() { return predict.CollectParams(); }

  void Split(const std::string &layer, ModelUtil &firstModel,
             ModelUtil &secondModel, bool force_cpu, bool inclusive = true);
  void CopyTrain(const std::string &layer, int out_size,
                 ModelUtil &train) const;
  void CopyTest(ModelUtil &test) const;
  void CopyDeploy(ModelUtil &deploy, Workspace &workspace) const;

  size_t Write(const std::string &path_prefix) const;
  size_t Read(const std::string &path_prefix);
  size_t WriteBundle(const std::string &filename) const;
  size_t ReadBundle(const std::string &filename);
  size_t WriteMeta(const std::string &filename) const;
  size_t ReadMeta(const std::string &filename);
  void SetName(const std::string &name);
  void SetDeviceCUDA();
  std::string Short();
  std::string Proto();

  void input_dims(const std::vector<int> &dims);
  std::vector<int> input_dims();
  void output_labels(const std::vector<std::string> &labels);
  std::vector<std::string> output_labels();

 public:
  NetUtil init;
  NetUtil predict;
  ModelMeta *meta;
};

}  // namespace caffe2

#endif  // UTIL_MODEL_H
