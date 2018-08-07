#ifndef UTIL_NET_H
#define UTIL_NET_H

#include <caffe2/core/operator.h>

namespace caffe2 {

class NetUtil {
 public:
  NetUtil(NetDef& net, const std::string& name = "") : net(net) {
    if (name.size()) {
      SetName(name);
    }
  }

  OperatorDef* AddOp(const std::string& name,
                     const std::vector<std::string>& inputs,
                     const std::vector<std::string>& outputs);

  OperatorDef* AddCreateDbOp(const std::string& reader,
                             const std::string& db_type,
                             const std::string& db_path);
  OperatorDef* AddTensorProtosDbInputOp(const std::string& reader,
                                        const std::string& data,
                                        const std::string& label,
                                        int batch_size);
  OperatorDef* AddTensorProtosDbInputOp(const std::string& reader,
										const std::vector<std::string>& outputs,
										int batch_size);
  OperatorDef* AddCoutOp(const std::vector<std::string>& params);
  OperatorDef* AddZeroOneOp(const std::string& pred, const std::string& label);
  OperatorDef* AddShowWorstOp(const std::string& pred, const std::string& label,
                              const std::string& data, float scale = 1.0,
                              float mean = 128.0);
  OperatorDef* AddTimePlotOp(const std::string& data,
                             const std::string& iter = "",
                             const std::string& window = "",
                             const std::string& label = "", int step = 0);
  OperatorDef* AddEnsureCpuOutputOp(const std::string& input,
                                    const std::string& output);
  OperatorDef* AddCopyFromCpuInputOp(const std::string& input,
                                     const std::string& output);
  OperatorDef* AddCopyOp(const std::string& input, const std::string& output);
  OperatorDef* AddCopyGPUToCPUOp(const std::string& input, const std::string& output);
  OperatorDef* AddCopyCPUToGPUOp(const std::string& input, const std::string& output);
  OperatorDef* AddCreateMutexOp(const std::string& param);
  OperatorDef* AddPrintOp(const std::string& param, bool to_file = false);
  OperatorDef* AddSummarizeOp(const std::string& param, bool to_file = false);

  OperatorDef* AddConstantFillOp(const std::vector<int>& shape,
                                 const std::string& output);
  OperatorDef* AddXavierFillOp(const std::vector<int>& shape,
                               const std::string& param);
  OperatorDef* AddMSRAFillOp(const std::vector<int>& shape,
                             const std::string& param);
  OperatorDef* AddUniformFillOp(const std::vector<int>& shape, float min,
                                float max, const std::string& param);
  OperatorDef * AddGaussianFillOp(const std::vector<int>& shape, float mean,
										float std, const std::string& param);
  OperatorDef* AddConstantFloatFillOp(const std::vector<int>& shape, float value,
                                 const std::string& output);
  OperatorDef* AddConstantIntFillOp(const std::vector<int>& shape, int value,
								const std::string& output);
  OperatorDef* AddConstantLongFillOp(const std::vector<int>& shape, int64_t value,
                                 const std::string& output);
  OperatorDef* AddConstantFillWithOp(float value, const std::string& input,
                                     const std::string& output);
  OperatorDef* AddVectorFillOp(const std::vector<int>& values,
                               const std::string& name);
  OperatorDef* AddGivenTensorFillOp(const TensorCPU& tensor,
                                    const std::string& name);
  OperatorDef* AddTransposeOp(const std::string& input, const std::string& output,
							  const std::vector<int>& axes);
  OperatorDef* AddFlattenOp(const std::string& input, const std::string& output,
							int axis = 1);

  OperatorDef* AddConvOp(const std::string& input, const std::string& w,
                         const std::string& b, const std::string& output,
                         int stride, int padding, int kernel, int group = 0,
                         const std::string& order = "NCHW");

  OperatorDef* AddConv1DOp(const std::string& input, const std::string& w,
		  				const std::string& b, const std::string& output,
						std::vector<int> strides, std::vector<int> pads, std::vector<int> kernels);

  OperatorDef* AddConv3DOp(const std::string& input, const std::string& w,
						const std::string& b, const std::string& output,
						std::vector<int> strides, std::vector<int> pads, std::vector<int> kernels);
  OperatorDef* AddConvTransposeOp(const std::string& input, const std::string& w,
                         const std::string& b, const std::string& output,
                         int stride, int padding, int kernel,
                         const std::string& order = "NCHW");
  OperatorDef* AddReluOp(const std::string& input, const std::string& output);
  OperatorDef * AddLeakyReluOp(const std::string& input,
								const std::string& output, float alpha);
  OperatorDef * AddSigmoidOp(const std::string& input,
								const std::string& output);
  OperatorDef * AddTanhOp(const std::string& input,
                                const std::string& output);
  OperatorDef* AddLrnOp(const std::string& input, const std::string& output,
                        int size, float alpha, float beta, float bias,
                        const std::string& order = "NCHW");
  OperatorDef* AddMaxPoolOp(const std::string& input, const std::string& output,
                            int stride, int padding, int kernel,
                            const std::string& order = "NCHW");
  OperatorDef* AddGlobalMaxPoolOp(const std::string& input,
		            const std::string& output, const std::string& order = "NCHW");
  OperatorDef* AddMaxPool1DOp(const std::string& input,
		            const std::string& output, vector<int> strides,
			    vector<int> pads,vector<int> kernels,
			    const std::string& order = "NCHW");
  OperatorDef* AddGlobalMaxPool1DOp(const std::string& input,
		            const std::string& output,const std::string& order = "NCHW");
  OperatorDef* AddMaxPoolWithIndexOp(const std::string& input, const std::string& output, const std::string& index,
                            std::vector<int> strides, std::vector<int> pads, std::vector<int> kernels,
                            const std::string& order = "NCHW");
  OperatorDef* AddAveragePoolOp(const std::string& input,
                                const std::string& output, int stride,
                                int padding, int kernel,
                                const std::string& order = "NCHW");
  OperatorDef* AddFcOp(const std::string& input, const std::string& w,
                       const std::string& b, const std::string& output,
                       int axis = 1);
  OperatorDef* AddDropoutOp(const std::string& input, const std::string& output,
                            float ratio, bool test = false);
  OperatorDef* AddSoftmaxOp(const std::string& input, const std::string& output,
                            int axis = 1);
  OperatorDef* AddConcatOp(const std::vector<std::string>& inputs,
                           const std::string& output,
                           const std::string& order = "NCHW");
  OperatorDef* AddConcatOp(const std::vector<std::string>& inputs,
                           const std::string& output, int axis);
  OperatorDef* AddSpatialBNOp(const std::vector<std::string>& inputs,
                              const std::vector<std::string>& outputs,
                              float epsilon = 1e-5f, float momentum = 0.9,
                              bool test = false,
                              const std::string& order = "NCHW");
  OperatorDef* AddMulOp(const std::vector<std::string>& inputs,
                        const std::string& output, int broadcast = 1,
                        int axis = 1);
  OperatorDef* AddDivOp(const std::vector<std::string>& inputs,
						const std::string& output, int broadcast = 1,
						int axis = 1);
  OperatorDef* AddAddOp(const std::vector<std::string>& inputs,
                        const std::string& output, int broadcast = 1,
                        int axis = 1);
  OperatorDef* AddModOp(const std::string& input,
						const std::string& output, int divisor);
  OperatorDef* AddSquaredL2DistanceOp(const std::vector<std::string> & inputs,
						const std::string& output);
  OperatorDef* AddL1DistanceOp(const std::vector<std::string> & inputs,
                        const std::string& output);
  OperatorDef* AddPowOp(const std::string & input, const std::string & output, float exponent);
  OperatorDef* AddSubOp(const std::vector<std::string> & inputs, const std::string & output, int broadcast = 1, int axis = 1);
  OperatorDef* AddUpsampleNearestOp(const std::string & input,const std::string & output, int scale = 2);

  OperatorDef* AddLSTMUnitOp(const std::vector<std::string>& inputs,
                             const std::vector<std::string>& outputs,
                             int drop_states = 0, float forget_bias = 0.f);
  OperatorDef* AddRecurrentNetworkOp(const std::string& seq_lengths,
                                     const std::string& hidden_init,
                                     const std::string& cell_init,
                                     const std::string& scope,
                                     const std::string& hidden_output,
                                     const std::string& cell_state,
                                     bool force_cpu);

  OperatorDef* AddAccuracyOp(const std::string& pred, const std::string& label,
                             const std::string& accuracy, int top_k = 0);
  OperatorDef* AddLabelCrossEntropyOp(const std::string& pred,
                                      const std::string& label,
                                      const std::string& xent);
  OperatorDef* AddSigmoidCrossEntropyWithLogits(const std::string& logits,
		  				const std::string& label,
						const std::string& xent);
  OperatorDef* AddAveragedLossOp(const std::string& input,
                                 const std::string& loss);
  OperatorDef* AddSoftmaxWithLossOp(const std::vector<std::string>& inputs,
								const std::vector<std::string>& outputs, int axis = 1);
  OperatorDef* AddDiagonalOp(const std::string& input,
                             const std::string& diagonal,
                             const std::vector<int>& offset);
  OperatorDef* AddSquaredL2Op(const std::string& input, const std::string& l2);
  OperatorDef* AddSquaredL2ChannelOp(const std::string& input,
                                     const std::string& l2, int channel);
  OperatorDef* AddBackMeanOp(const std::string& input, const std::string& mean,
                             int count = 1);
  OperatorDef* AddReduceBackMeanOp(const std::string& input, const std::string& mean);
  OperatorDef* AddReduceBackSumOp(const std::string& input, const std::string& sum);
  OperatorDef* AddReduceTailSumOp(const std::string& input, const std::string& sum);
  OperatorDef* AddMeanStdevOp(const std::string& input, const std::string& mean,
                              const std::string& scale);
  OperatorDef* AddAffineScaleOp(const std::string& input,
                                const std::string& mean,
                                const std::string& scale,
                                const std::string& transformed,
                                bool inverse = false);
  OperatorDef* AddSliceOp(const std::string& input, const std::string& output,
                          const std::vector<std::pair<int, int>>& ranges);
  OperatorDef* AddReshapeOp(const std::string& input, const std::string& output,
                            const std::vector<int>& shape);

  OperatorDef* AddWeightedSumOp(const std::vector<std::string>& inputs,
                                const std::string& sum);
  OperatorDef* AddSumOp(const std::vector<std::string>& inputs,
                        const std::string& sum);
  OperatorDef* AddMomentumSgdOp(const std::string& param,
                                const std::string& moment,
                                const std::string& grad, const std::string& lr);
  OperatorDef* AddAdagradOp(const std::string& param, const std::string& moment,
                            const std::string& grad, const std::string& lr);
  OperatorDef* AddAdamOp(const std::string& param,
                         const std::vector<std::string>& moments,
                         const std::string& grad, const std::string& lr,
                         const std::string& iter);
  OperatorDef* AddRmsPropOp(const std::string& grad, const std::string& meansq,
                            const std::string& mom, const std::string& lr,
                            float decay = 0.9f, float momentum = 0.8f,
                            float epsilon = 1e-5f);
  OperatorDef* AddScaleOp(const std::string& input, const std::string& output,
                          float scale);
  OperatorDef* AddClipOp(const std::string& input, const std::string& output,
                         float min, float max);
  OperatorDef* AddCastOp(const std::string& input, const std::string& output,
                         TensorProto::DataType type);
  OperatorDef* AddStopGradientOp(const std::string& param);
  OperatorDef* AddSaveOp(const std::vector<std::string>& inputs,const std::string& type,
						const std::string & path);
  OperatorDef* AddLoadOp(const std::vector<std::string>& outputs,const std::string& type,
						const std::string & path);
  OperatorDef* AddIterOp(const std::string& iter);
  OperatorDef* AddAtomicIterOp(const std::string& mutex,
                               const std::string& iter);
  OperatorDef* AddLearningRateOp(const std::string& iter,
                                 const std::string& rate, float base_rate,
                                 float gamma = 0.999f, int stepsize = 1);
  OperatorDef* AddCheckpointOp(const std::vector<std::string>& inputs,
                               int every, const std::string& db_type,
                               const std::string& db);

  void AddInput(const std::string input);
  void AddOutput(const std::string output);
  const std::string& Input(int i);
  const std::string& Output(int i);

  void SetName(const std::string name);
  void SetType(const std::string type);

  void SetFillToTrain();
  void SetRenameInplace();
  void SetEngineOps(const std::string engine);

  OperatorDef* AddGradientOp(OperatorDef& op);
  OperatorDef* AddGradientOps(
  OperatorDef& op, std::map<std::string, std::pair<int, int>>& split_inputs,
      std::map<std::string, std::string>& pass_replace,
	  std::set<std::string>& stop_inputs);
  void AddGradientOps();
  void AddGradientOps(NetUtil& target) const;

  std::map<std::string, int> CollectParamSizes();
  std::vector<std::string> CollectParams();
  std::vector<OperatorDef> CollectGradientOps(
      std::map<std::string, std::pair<int, int>>& split_inputs) const;
  std::set<std::string> CollectLayers(const std::string& layer,
                                      bool forward = false);

  void CheckLayerAvailable(const std::string& layer);
  std::string Proto();
  std::string Short();
  void Print();
  size_t Write(const std::string& path) const;
  size_t WriteText(const std::string& path) const;
  size_t WriteGraph(const std::string& path) const;
  size_t Read(const std::string& path);

  void SetDeviceCUDA();

 public:
  NetDef& net;
};

}  // namespace caffe2

#endif  // UTIL_NET_H
