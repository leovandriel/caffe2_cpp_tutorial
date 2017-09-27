#ifndef ZOO_KEEPER_H
#define ZOO_KEEPER_H

#include "caffe2/util/net.h"
#include "caffe2/zoo/alexnet.h"
#include "caffe2/zoo/googlenet.h"
#include "caffe2/zoo/resnet.h"
#include "caffe2/zoo/squeezenet.h"
#include "caffe2/zoo/vgg.h"

#ifdef WITH_CURL
#include <curl/curl.h>
#endif

namespace caffe2 {

const int bar_length = 64;

const std::map<std::string, std::map<std::string, std::string>>
    keeper_model_lookup{
        {"alexnet",
         {{"res/alexnet_predict_net.pb",
           "https://github.com/leonardvandriel/caffe2_models/raw/master/model/"
           "alexnet_predict_net.pb"},
          {"res/alexnet_init_net.pb",
           "https://github.com/leonardvandriel/caffe2_models/raw/master/model/"
           "alexnet_init_net.pb"}}},
        {"googlenet",
         {
             {"res/googlenet_predict_net.pb",
              "https://github.com/leonardvandriel/caffe2_models/raw/master/"
              "model/googlenet_predict_net.pb"},
             {"res/googlenet_init_net.pb",
              "https://github.com/leonardvandriel/caffe2_models/raw/master/"
              "model/googlenet_init_net.pb"},
         }},
        {"squeezenet",
         {
             {"res/squeezenet_predict_net.pb",
              "https://github.com/caffe2/models/raw/master/squeezenet/"
              "predict_net.pb"},
             {"res/squeezenet_init_net.pb",
              "https://github.com/caffe2/models/raw/master/squeezenet/"
              "init_net.pb"},
         }},
        {"vgg16",
         {{"res/vgg16_predict_net.pb",
           "https://github.com/leonardvandriel/caffe2_models/raw/master/model/"
           "vgg16_predict_net.pb"},
          {"res/vgg16_init_net.pb",
           "https://github.com/leonardvandriel/caffe2_models/raw/master/model/"
           "vgg16_init_net.pb"}}},
        {"vgg19",
         {{"res/vgg19_predict_net.pb",
           "https://github.com/leonardvandriel/caffe2_models/raw/master/model/"
           "vgg19_predict_net.pb"},
          {"res/vgg19_init_net.pb",
           "https://github.com/leonardvandriel/caffe2_models/raw/master/model/"
           "vgg19_init_net.pb"}}},
        {"resnet50",
         {{"res/resnet50_predict_net.pb",
           "https://github.com/leonardvandriel/caffe2_models/raw/master/model/"
           "resnet50_predict_net.pb"},
          {"res/resnet50_init_net.pb",
           "https://github.com/leonardvandriel/caffe2_models/raw/master/model/"
           "resnet50_init_net.pb"}}},
        {"resnet101",
         {{"res/resnet101_predict_net.pb",
           "https://github.com/leonardvandriel/caffe2_models/raw/master/model/"
           "resnet101_predict_net.pb"},
          {"res/resnet101_init_net.pb",
           "https://github.com/leonardvandriel/caffe2_models/raw/master/model/"
           "resnet101_init_net.pb"}}},
        {"resnet152",
         {{"res/resnet152_predict_net.pb",
           "https://github.com/leonardvandriel/caffe2_models/raw/master/model/"
           "resnet152_predict_net.pb"},
          {"res/resnet152_init_net.pb",
           "https://github.com/leonardvandriel/caffe2_models/raw/master/model/"
           "resnet152_init_net.pb"}}}};

int keeper_progress_func(const char *filename, double total_down,
                         double current_down, double total_up,
                         double current_up) {
  int length = 72;
  if (total_down) {
    int prom = 1000 * current_down / total_down;
    int bar = bar_length * current_down / total_down;
    std::cerr << '\r' << std::string(bar, '#')
              << std::string(bar_length - bar, ' ') << std::setw(5)
              << ((float)prom / 10) << "%"
              << " " << filename << std::flush;
  }
  return 0;
}

class Keeper {
 public:
  Keeper(const std::string name) : name_(name) {}

  bool download(const std::string &filename, const std::string &url) {
#ifdef WITH_CURL
    FILE *fp = fopen(filename.c_str(), "wb");
    CURL *curl = curl_easy_init();
    if (!curl) {
      return false;
    }
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_NOPROGRESS, false);
    curl_easy_setopt(curl, CURLOPT_PROGRESSFUNCTION, keeper_progress_func);
    curl_easy_setopt(curl, CURLOPT_PROGRESSDATA, filename.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, fwrite);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);
    CURLcode result = curl_easy_perform(curl);
    curl_easy_cleanup(curl);
    fclose(fp);
    std::cerr << '\r' << std::string(filename.length() + bar_length + 7, ' ')
              << '\r';
    return result == CURLE_OK;
#else
    std::cout << "model download not supported, install cURL" << std::endl;
    return false;
#endif
  }

  bool ensureFile(const std::string &filename, const std::string &url) {
    if (std::ifstream(filename).good()) {
      return true;
    }
    return download(filename, url);
  }

  bool ensureModel() {
    if (keeper_model_lookup.find(name_) == keeper_model_lookup.end()) {
      return false;
    }
    const std::map<std::string, std::string> &pairs =
        keeper_model_lookup.at(name_);
    for (const auto &pair : pairs) {
      if (!ensureFile(pair.first, pair.second)) {
        return false;
      }
    }
    return true;
  }

  void addTrainedModel(NetDef &init_model, NetDef &predict_model) {
    CAFFE_ENFORCE(ensureModel(), "model ", name_, " not found");
    std::string init_filename = "res/" + name_ + "_init_net.pb";
    std::string predict_filename = "res/" + name_ + "_predict_net.pb";
    CAFFE_ENFORCE(ReadProtoFromFile(init_filename.c_str(), &init_model));
    CAFFE_ENFORCE(ReadProtoFromFile(predict_filename.c_str(), &predict_model));
  }

  void addUntrainedModel(NetDef &init_model, NetDef &predict_model) {
    if (name_ == "alexnet") {
      AlexNetModel(init_model, predict_model).Add();
    } else if (name_ == "googlenet") {
      GoogleNetModel(init_model, predict_model).Add();
    } else if (name_ == "squeezenet") {
      SqueezeNetModel(init_model, predict_model).Add();
    } else if (name_ == "vgg16") {
      VGGModel(init_model, predict_model).Add(16);
    } else if (name_ == "vgg19") {
      VGGModel(init_model, predict_model).Add(19);
    } else if (name_ == "resnet50") {
      ResNetModel(init_model, predict_model).Add(50);
    } else if (name_ == "resnet101") {
      ResNetModel(init_model, predict_model).Add(101);
    } else if (name_ == "resnet152") {
      ResNetModel(init_model, predict_model).Add(152);
    } else {
      std::cerr << "model " << name_ << " not implemented" << std::endl;
    }
  }

  void AddModel(NetDef &init_model, NetDef &predict_model, bool trained) {
    if (trained) {
      addTrainedModel(init_model, predict_model);
    } else {
      addUntrainedModel(init_model, predict_model);
    }
  }

 protected:
  const std::string name_;
};

}  // namespace caffe2

#endif  // ZOO_KEEPER_H
