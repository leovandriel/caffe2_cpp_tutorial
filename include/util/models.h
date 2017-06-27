#ifndef MODELS_H
#define MODELS_H

#include <curl/curl.h>

namespace caffe2 {

const int bar_length = 64;

const std::map<std::string, std::map<std::string, std::string>> model_lookup {
  { "alexnet", {
    { "res/alexnet_predict_net.pb", "https://s3.amazonaws.com/caffe2/models/bvlc_alexnet/predict_net.pb" },
    { "res/alexnet_init_net.pb", "https://s3.amazonaws.com/caffe2/models/bvlc_alexnet/init_net.pb" }
  }},
  { "googlenet", {
    { "res/googlenet_predict_net.pb", "https://s3.amazonaws.com/caffe2/models/bvlc_googlenet/predict_net.pb" },
    { "res/googlenet_init_net.pb", "https://s3.amazonaws.com/caffe2/models/bvlc_googlenet/init_net.pb" },
  }},
  { "squeezenet", {
    { "res/squeezenet_predict_net.pb", "https://s3.amazonaws.com/caffe2/models/squeezenet/predict_net.pb" },
    { "res/squeezenet_init_net.pb", "https://s3.amazonaws.com/caffe2/models/squeezenet/init_net.pb" },
  }},
  { "vgg16", {
    { "res/vgg16_predict_net.pb", "https://github.com/leonardvandriel/caffe2_models/raw/master/model/vgg16_predict_net.pb" },
    { "res/vgg16_init_net.pb", "https://github.com/leonardvandriel/caffe2_models/raw/master/model/vgg16_init_net.pb" }
  }},
  { "vgg19", {
    { "res/vgg19_predict_net.pb", "https://github.com/leonardvandriel/caffe2_models/raw/master/model/vgg19_predict_net.pb" },
    { "res/vgg19_init_net.pb", "https://github.com/leonardvandriel/caffe2_models/raw/master/model/vgg19_init_net.pb" }
  }},
  { "resnet50", {
    { "res/resnet50_predict_net.pb", "https://github.com/leonardvandriel/caffe2_models/raw/master/model/resnet50_predict_net.pb" },
    { "res/resnet50_init_net.pb", "https://github.com/leonardvandriel/caffe2_models/raw/master/model/resnet50_init_net.pb" }
  }},
  { "resnet101", {
    { "res/resnet101_predict_net.pb", "https://github.com/leonardvandriel/caffe2_models/raw/master/model/resnet101_predict_net.pb" },
    { "res/resnet101_init_net.pb", "https://github.com/leonardvandriel/caffe2_models/raw/master/model/resnet101_init_net.pb" }
  }},
  { "resnet152", {
    { "res/resnet152_predict_net.pb", "https://github.com/leonardvandriel/caffe2_models/raw/master/model/resnet152_predict_net.pb" },
    { "res/resnet152_init_net.pb", "https://github.com/leonardvandriel/caffe2_models/raw/master/model/resnet152_init_net.pb" }
  }}
};

int progress_func(const char* filename, double total_down, double current_down, double total_up, double current_up)
{
  int length = 72;
  if (total_down) {
    int prom = 1000 * current_down / total_down;
    int bar = bar_length * current_down / total_down;
    std::cerr << '\r' << std::string(bar, '#') << std::string(bar_length - bar, ' ') << std::setw(5) << ((float)prom / 10) << "%" << " " << filename << std::flush;
  }
  return 0;
}

bool download(const std::string &filename, const std::string &url) {
  FILE *fp = fopen(filename.c_str(), "wb");
  CURL *curl = curl_easy_init();
  if (!curl) {
    return false;
  }
  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
  curl_easy_setopt(curl, CURLOPT_NOPROGRESS, false);
  curl_easy_setopt(curl, CURLOPT_PROGRESSFUNCTION, progress_func);
  curl_easy_setopt(curl, CURLOPT_PROGRESSDATA, filename.c_str());
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, fwrite);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);
  CURLcode result = curl_easy_perform(curl);
  curl_easy_cleanup(curl);
  fclose(fp);
  std::cerr << '\r' << std::string(filename.length() + bar_length + 7, ' ') << '\r';
  return result == CURLE_OK;
}

bool ensureFile(const std::string &filename, const std::string &url) {
  if (std::ifstream(filename).good()) {
    return true;
  }
  return download(filename, url);
}

bool ensureModel(const std::string &name) {
  if (model_lookup.find(name) == model_lookup.end()) {
    return false;
  }
  const std::map<std::string, std::string> &pairs = model_lookup.at(name);
  for (const auto &pair: pairs) {
    if(!ensureFile(pair.first, pair.second)) {
      return false;
    }
  }
  return true;
}

}  // namespace caffe2

#endif  // MODELS_H
