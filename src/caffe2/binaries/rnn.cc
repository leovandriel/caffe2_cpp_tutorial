#include <caffe2/core/init.h>
#include "caffe2/util/blob.h"
#include "caffe2/util/model.h"
#include "caffe2/util/net.h"

#include "caffe2/util/cmd.h"

CAFFE2_DEFINE_string(model, "char_rnn", "The RNN model.");
CAFFE2_DEFINE_string(train_data, "res/shakespeare.txt",
                     "Path to training data in a text file format");

CAFFE2_DEFINE_int(iters, 10 * 1000, "The of training runs.");
CAFFE2_DEFINE_int(seq_length, 25, "One training example sequence length");
CAFFE2_DEFINE_int(batch, 1, "Training batch size");
CAFFE2_DEFINE_int(iters_to_report, 500,
                  "How often to report loss and generate text");
CAFFE2_DEFINE_int(hidden_size, 100, "Dimension of the hidden representation");
CAFFE2_DEFINE_int(gen_length, 500, "One forward example sequence length");

namespace caffe2 {

void AddFC(ModelUtil &model, const std::string &input,
           const std::string &output, int in_size, int out_size) {
  model.init.AddXavierFillOp({out_size, in_size}, output + "_w");
  model.predict.AddInput(output + "_w");
  model.init.AddConstantFillOp({out_size}, output + "_b");
  model.predict.AddInput(output + "_b");
  model.predict.AddFcOp(input, output + "_w", output + "_b", output, 2)
      ->set_engine("CUDNN");
}

void AddLSTM(ModelUtil &model, const std::string &input_blob,
             const std::string &seq_lengths, const std::string &hidden_init,
             const std::string &cell_init, int vocab_size, int hidden_size,
             const std::string &scope, std::string *hidden_output,
             std::string *cell_state) {
  *hidden_output = scope + "/hidden_t_last";
  *cell_state = scope + "/cell_t_last";
  AddFC(model, input_blob, scope + "/i2h", vocab_size, 4 * hidden_size);
  // sight hack
  model.init.AddXavierFillOp({4 * hidden_size, hidden_size},
                             scope + "/gates_t_w");
  model.predict.AddInput(scope + "/gates_t_w");
  model.init.AddConstantFillOp({4 * hidden_size}, scope + "/gates_t_b");
  model.predict.AddInput(scope + "/gates_t_b");
  model.predict.AddRecurrentNetworkOp(seq_lengths, hidden_init, cell_init,
                                      scope, *hidden_output, *cell_state,
                                      FLAGS_device == "cpu");
}

void AddSGD(ModelUtil &model, float base_learning_rate,
            const std::string &policy, int stepsize, float gamma) {
  model.predict.AddAtomicIterOp("iteration_mutex", "optimizer_iteration")
      ->mutable_device_option()
      ->set_device_type(CPU);
  model.init.AddConstantFillOp({1}, (int64_t)0, "optimizer_iteration")
      ->mutable_device_option()
      ->set_device_type(CPU);
  model.init.AddCreateMutexOp("iteration_mutex")
      ->mutable_device_option()
      ->set_device_type(CPU);
  model.predict.AddInput("iteration_mutex");
  model.predict.AddInput("optimizer_iteration");
  model.init.AddConstantFillOp({1}, 1.f, "ONE");
  model.predict.AddInput("ONE");
  model.predict.AddLearningRateOp("optimizer_iteration", "lr",
                                  base_learning_rate, gamma);
  std::vector<std::string> params({"LSTM/gates_t_w", "LSTM/i2h_b",
                                   "char_rnn_blob_0_w", "char_rnn_blob_0_b",
                                   "LSTM/gates_t_b", "LSTM/i2h_w"});
  for (auto &param : params) {
    model.predict.AddWeightedSumOp({param, "ONE", param + "_grad", "lr"},
                                   param);
  }
}

void run() {
  std::cout << std::endl;
  std::cout << "## Caffe2 RNNs and LSTM Tutorial ##" << std::endl;
  std::cout << "https://caffe2.ai/docs/RNNs-and-LSTM-networks.html"
            << std::endl;
  std::cout << std::endl;

  if (!std::ifstream(FLAGS_train_data).good()) {
    std::cerr << "error: Text file missing: " << FLAGS_train_data << std::endl;
    std::cerr << "Make sure to first run ./script/download_resource.sh"
              << std::endl;
    return;
  }

  auto cuda = (FLAGS_device != "cpu" && cmd_setup_cuda());

  std::cout << "model: " << FLAGS_model << std::endl;
  std::cout << "train-data: " << FLAGS_train_data << std::endl;
  std::cout << "iters: " << FLAGS_iters << std::endl;
  std::cout << "seq-length: " << FLAGS_seq_length << std::endl;
  std::cout << "batch: " << FLAGS_batch << std::endl;
  std::cout << "iters-to-report: " << FLAGS_iters_to_report << std::endl;
  std::cout << "hidden-size: " << FLAGS_hidden_size << std::endl;
  std::cout << "gen-length: " << FLAGS_gen_length << std::endl;

  std::cout << "device: " << FLAGS_device << std::endl;
  std::cout << "using cuda: " << (cuda ? "true" : "false") << std::endl;
  ;
  std::cout << "dump-model: " << (FLAGS_dump_model ? "true" : "false")
            << std::endl;

  std::cout << std::endl;

  // >>> with open(args.train_data) as f: self.text = f.read()
  std::ifstream infile(FLAGS_train_data);
  std::stringstream buffer;
  buffer << infile.rdbuf();
  auto text = buffer.str();

  if (!text.size()) {
    std::cerr << "unable to read input text" << std::endl;
    return;
  }

  // >>> self.vocab = list(set(self.text))
  std::set<char> vocab_set(text.begin(), text.end());
  std::vector<char> vocab(vocab_set.begin(), vocab_set.end());

  // >>> self.char_to_idx = {ch: idx for idx, ch in enumerate(self.vocab)}
  // >>> self.idx_to_char = {idx: ch for idx, ch in enumerate(self.vocab)}
  std::map<char, int> char_to_idx;
  std::map<int, char> idx_to_char;
  auto index = 0;
  for (auto c : vocab) {
    char_to_idx[c] = index;
    idx_to_char[index++] = c;
    // std::cout << c;
  }

  // >>> self.D = len(self.char_to_idx)
  auto D = (int)char_to_idx.size();

  // >>> print("Input has {} characters. Total input size:
  // {}".format(len(self.vocab), len(self.text)))
  std::cout << "Input has " << vocab.size()
            << " characters. Total input size: " << text.size() << std::endl;

  // >>> log.debug("Start training")
  std::cout << "Start training" << std::endl;

  // >>> model = model_helper.ModelHelper(name="char_rnn")
  NetDef init_model, predict_model;
  ModelUtil model(init_model, predict_model, "char_rnn");

  // >>> input_blob, seq_lengths, hidden_init, cell_init, target =
  // model.net.AddExternalInputs('input_blob', 'seq_lengths', 'hidden_init',
  // 'cell_init', 'target')
  model.predict.AddInput("input_blob");
  model.predict.AddInput("seq_lengths");
  model.predict.AddInput("hidden_init");
  model.predict.AddInput("cell_init");
  model.predict.AddInput("target");

  // >>> hidden_output_all, self.hidden_output, _, self.cell_state = LSTM(model,
  // input_blob, seq_lengths, (hidden_init, cell_init), self.D,
  // self.hidden_size, scope="LSTM")
  std::string hidden_output;
  std::string cell_state;
  AddLSTM(model, "input_blob", "seq_lengths", "hidden_init", "cell_init", D,
          FLAGS_hidden_size, "LSTM", &hidden_output, &cell_state);

  // >>> output = brew.fc(model, hidden_output_all, None,
  // dim_in=self.hidden_size, dim_out=self.D, axis=2)
  AddFC(model, "LSTM/hidden_t_all", "char_rnn_blob_0", FLAGS_hidden_size, D);

  // >>> softmax = model.net.Softmax(output, 'softmax', axis=2)
  model.predict.AddSoftmaxOp("char_rnn_blob_0", "softmax", 2);

  // >>> softmax_reshaped, _ = model.net.Reshape(softmax, ['softmax_reshaped',
  // '_'], shape=[-1, self.D])
  model.predict.AddReshapeOp("softmax", "softmax_reshaped", {-1, D});

  // >>> self.forward_net = core.Net(model.net.Proto())
  NetDef train_model(model.predict.net);
  NetUtil train(train_model, "train");

  // >>> xent = model.net.LabelCrossEntropy([softmax_reshaped, target], 'xent')
  train.AddLabelCrossEntropyOp("softmax_reshaped", "target", "xent");

  // >>> loss = model.net.AveragedLoss(xent, 'loss')
  train.AddAveragedLossOp("xent", "loss");

  // >>> model.AddGradientOperators([loss])
  train.AddConstantFillWithOp(1.f, "loss", "loss_grad");
  train.AddGradientOps();

  // >>> build_sgd(model, base_learning_rate=0.1 * self.seq_length,
  // policy="step", stepsize=1, gamma=0.9999)
  ModelUtil t(model.init, train);
  AddSGD(t, 0.1 * FLAGS_seq_length, "step", 1, 0.9999);

  // >>> self.model = model
  // >>> self.predictions = softmax
  // >>> self.loss = loss
  auto predictions = "softmax";
  auto loss_name = "loss";

  // >>> self.prepare_state = core.Net("prepare_state")
  // >>> self.prepare_state.Copy(self.hidden_output, hidden_init)
  // >>> self.prepare_state.Copy(self.cell_state, cell_init)
  NetDef prepare_model;
  NetUtil prepare(prepare_model, "prepare_state");
  prepare.AddCopyOp(hidden_output, "hidden_init");
  prepare.AddCopyOp(cell_state, "cell_init");
  prepare.AddInput(hidden_output);
  prepare.AddInput(cell_state);

  if (FLAGS_device != "cpu") {
    model.SetDeviceCUDA();
    train.SetDeviceCUDA();
    prepare.SetDeviceCUDA();
  }

  if (FLAGS_dump_model) {
    std::cout << model.init.Short();
    std::cout << train.Short();
    std::cout << prepare.Short();
  }

  // >>> from caffe2.python import core, workspace, model_helper, utils, brew
  Workspace workspace("tmp");

  // >>> log.debug("Training model")
  std::cout << "Train model" << std::endl;

  // >>> workspace.RunNetOnce(self.model.param_init_net)
  CAFFE_ENFORCE(workspace.RunNetOnce(model.init.net));

  // >>> smooth_loss = -np.log(1.0 / self.D) * self.seq_length
  auto smooth_loss = -log(1.0 / D) * FLAGS_seq_length;
  // >>> last_n_iter = 0
  auto last_n_iter = 0;
  // >>> last_n_loss = 0.0
  auto last_n_loss = 0.f;
  // >>> num_iter = 0
  auto num_iter = 0;
  // >>> N = len(self.text)
  auto N = text.size();

  // >>> text_block_positions = np.zeros(self.batch_size, dtype=np.int32)
  std::vector<int> text_block_positions(FLAGS_batch);
  // >>> text_block_size = N // self.batch_size
  auto text_block_size = N / FLAGS_batch;
  // >>> text_block_starts = list(range(0, N, text_block_size))
  std::vector<int> text_block_starts;
  for (auto i = 0; i < N; i += text_block_size) {
    text_block_starts.push_back(i);
  }
  // >>> text_block_sizes = [text_block_size] * self.batch_size
  std::vector<int> text_block_sizes(FLAGS_batch, text_block_size);
  // >>> text_block_sizes[self.batch_size - 1] += N % self.batch_size
  text_block_sizes[FLAGS_batch - 1] += N % FLAGS_batch;
  // >>> assert sum(text_block_sizes) == N
  CAFFE_ENFORCE_EQ(std::accumulate(text_block_sizes.begin(),
                                   text_block_sizes.end(), 0, std::plus<int>()),
                   N);

  // >>> workspace.FeedBlob(self.hidden_output, np.zeros([1, self.batch_size,
  // self.hidden_size], dtype=np.float32))
  {
    std::vector<float> data(FLAGS_batch * FLAGS_hidden_size);
    auto value = TensorCPU({1, FLAGS_batch, FLAGS_hidden_size}, data, NULL);
    BlobUtil(*workspace.CreateBlob(hidden_output)).Set(value, true);
  }
  // >>> workspace.FeedBlob(self.cell_state, np.zeros([1, self.batch_size,
  // self.hidden_size], dtype=np.float32))
  {
    std::vector<float> data(FLAGS_batch * FLAGS_hidden_size);
    auto value = TensorCPU({1, FLAGS_batch, FLAGS_hidden_size}, data, NULL);
    BlobUtil(*workspace.CreateBlob(cell_state)).Set(value, true);
  }
  // >>> workspace.CreateNet(self.prepare_state)
  CAFFE_ENFORCE(workspace.CreateNet(prepare.net));

  // >>> last_time = datetime.now()
  auto last_time = clock();
  // >>> progress = 0
  auto progress = 0;

  // >>> CreateNetOnce(self.model.net)
  workspace.CreateBlob("input_blob");
  workspace.CreateBlob("seq_lengths");
  workspace.CreateBlob("target");
  CAFFE_ENFORCE(workspace.CreateNet(train.net));

  // >>> CreateNetOnce(self.forward_net)
  CAFFE_ENFORCE(workspace.CreateNet(model.predict.net));

  // >>> while True:
  while (num_iter < FLAGS_iters) {
    // >>> workspace.FeedBlob("seq_lengths", np.array([self.seq_length] *
    // self.batch_size, dtype=np.int32))
    {
      std::vector<int> data(FLAGS_batch, FLAGS_seq_length);
      auto value = TensorCPU({FLAGS_batch}, data, NULL);
      BlobUtil(*workspace.CreateBlob("seq_lengths")).Set(value, true);
    }

    // >>> workspace.RunNet(self.prepare_state.Name())
    CAFFE_ENFORCE(workspace.RunNet(prepare.net.name()));

    // >>> input = np.zeros([self.seq_length, self.batch_size,
    // self.D]).astype(np.float32)
    std::vector<float> input(FLAGS_seq_length * FLAGS_batch * D);
    // >>> target = np.zeros([self.seq_length *
    // self.batch_size]).astype(np.int32)
    std::vector<int> target(FLAGS_seq_length * FLAGS_batch);

    // >>> for e in range(self.batch_size):
    for (auto e = 0; e < FLAGS_batch; e++) {
      // >>> for i in range(self.seq_length):
      for (auto i = 0; i < FLAGS_seq_length; i++) {
        // >>> pos = text_block_starts[e] + text_block_positions[e]
        auto pos = text_block_starts[e] + text_block_positions[e];
        // >>> input[i][e][self._idx_at_pos(pos)] = 1
        input[i * FLAGS_batch * D + e * D + char_to_idx[text[pos]]] = 1;
        // >>> target[i * self.batch_size + e] = self._idx_at_pos((pos + 1) % N)
        target[i * FLAGS_batch + e] = char_to_idx[text[(pos + 1) % N]];
        // >>> text_block_positions[e] = (text_block_positions[e] + 1) %
        // text_block_sizes[e]
        text_block_positions[e] =
            (text_block_positions[e] + 1) % text_block_sizes[e];
        // >>> progress += 1
        progress++;
      }
    }

    // >>> workspace.FeedBlob('input_blob', input)
    {
      auto value = TensorCPU({FLAGS_seq_length, FLAGS_batch, D}, input, NULL);
      BlobUtil(*workspace.CreateBlob("input_blob")).Set(value, true);
    }
    // >>> workspace.FeedBlob('target', target)
    {
      auto value = TensorCPU({FLAGS_seq_length * FLAGS_batch}, target, NULL);
      BlobUtil(*workspace.CreateBlob("target")).Set(value, true);
    }

    // >>> workspace.RunNet(self.model.net.Name())
    CAFFE_ENFORCE(workspace.RunNet(train.net.name()));
    // >>> num_iter += 1
    num_iter++;
    // >>> last_n_iter += 1
    last_n_iter++;

    // >>> if num_iter % self.iters_to_report == 0:
    if (num_iter % FLAGS_iters_to_report == 0) {
      // >>> new_time = datetime.now()
      auto new_time = clock();
      // >>> print("Characters Per Second: {}".format(int(progress / (new_time -
      // last_time).total_seconds())))
      std::cout << "Characters Per Second: "
                << ((size_t)progress * CLOCKS_PER_SEC / (new_time - last_time))
                << std::endl;
      // >>> print("Iterations Per Second: {}".format(int(self.iters_to_report /
      // (new_time - last_time).total_seconds())))
      std::cout << "Iterations Per Second: "
                << ((size_t)FLAGS_iters_to_report * CLOCKS_PER_SEC /
                    (new_time - last_time))
                << std::endl;

      // >>> last_time = new_time
      last_time = new_time;
      // >>> progress = 0
      progress = 0;

      // >>> print("{} Iteration {} {}".format('-' * 10, num_iter, '-' * 10))
      std::cout << "---------- Iteration " << num_iter << " ----------"
                << std::endl;
    }

    // >>> loss = workspace.FetchBlob(self.loss) * self.seq_length
    auto loss = BlobUtil(*workspace.GetBlob(loss_name)).Get().data<float>()[0] *
                FLAGS_seq_length;
    // >>> smooth_loss = 0.999 * smooth_loss + 0.001 * loss
    smooth_loss = 0.999 * smooth_loss + 0.001 * loss;
    // >>> last_n_loss += loss
    last_n_loss += loss;

    // >>> if num_iter % self.iters_to_report == 0:
    if (num_iter % FLAGS_iters_to_report == 0) {
      // >>> text = '' + ch
      std::stringstream text;
      auto ch = vocab[(int)(vocab.size() * (float)rand() / RAND_MAX)];
      text << ch;

      // >>> for _i in range(500):
      for (auto i = 0; i < FLAGS_gen_length; i++) {
        // >>> workspace.FeedBlob("seq_lengths", np.array([1] * self.batch_size,
        // dtype=np.int32))
        {
          std::vector<int> data(FLAGS_batch, 1);
          auto value = TensorCPU({FLAGS_batch}, data, NULL);
          BlobUtil(*workspace.CreateBlob("seq_lengths")).Set(value, true);
        }

        // >>> workspace.RunNet(self.prepare_state.Name())
        CAFFE_ENFORCE(workspace.RunNet(prepare.net.name()));

        // >>> input = np.zeros([1, self.batch_size, self.D]).astype(np.float32)
        std::vector<float> input(FLAGS_batch * D);
        // >>> input[0][0][self.char_to_idx[ch]] = 1
        input[char_to_idx[ch]] = 1;

        // >>> workspace.FeedBlob("input_blob", input)
        {
          auto value = TensorCPU({1, FLAGS_batch, D}, input, NULL);
          BlobUtil(*workspace.CreateBlob("input_blob")).Set(value, true);
        }
        // >>> workspace.RunNet(self.forward_net.Name())
        CAFFE_ENFORCE(workspace.RunNet(model.predict.net.name()));

        // >>> p = workspace.FetchBlob(self.predictions)
        auto p = BlobUtil(*workspace.GetBlob(predictions)).Get();

        // >>> next = np.random.choice(self.D, p=p[0][0])
        auto data = p.data<float>();
        // for (auto j = 0; j < vocab.size(); j++) if (data[j] > 0.1) std::cout
        // << idx_to_char[j] << ":" << data[j] << " "; std::cout << std::endl;
        auto r = (float)rand() / RAND_MAX;
        auto next = vocab.size() - 1;
        for (auto j = 0; j < vocab.size(); j++) {
          r -= data[j];
          if (r <= 0) {
            next = j;
            break;
          }
        }

        // >>> ch = self.idx_to_char[next]
        ch = idx_to_char[next];

        // >>> text += ch
        text << ch;
      }

      // print(text)
      std::cout << text.str() << std::endl;

      // >>> log.debug("Loss since last report: {}".format(last_n_loss /
      // last_n_iter))
      std::cout << "Loss since last report: " << (last_n_loss / last_n_iter)
                << std::endl;
      // >>> log.debug("Smooth loss: {}".format(smooth_loss))
      std::cout << "Smooth loss: " << smooth_loss << std::endl;

      // >>> last_n_loss = 0.0
      last_n_loss = 0.f;
      // >>> last_n_iter = 0
      last_n_iter = 0;
    }
  }
}

}  // namespace caffe2

int main(int argc, char **argv) {
  caffe2::GlobalInit(&argc, &argv);
  caffe2::run();
  google::protobuf::ShutdownProtobufLibrary();
  return 0;
}
