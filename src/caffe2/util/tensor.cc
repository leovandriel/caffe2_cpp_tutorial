#include "caffe2/util/tensor.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include "cvplot/cvplot.h"

namespace caffe2 {

const auto screen_width = 1600;
const auto window_padding = 4;

template <typename T>
cv::Mat to_image(const Tensor<CPUContext> &tensor, int index, float scale,
                 float mean, int type) {
  CAFFE_ENFORCE_EQ(tensor.ndim(), 4);
  auto count = tensor.dim(0), depth = tensor.dim(1), height = tensor.dim(2),
       width = tensor.dim(3);
  CAFFE_ENFORCE_LT(index, count);
  auto data = tensor.data<T>() + (index * width * height * depth);
  vector<cv::Mat> channels(depth);
  for (auto &j : channels) {
    j = cv::Mat(height, width, type, (void *)data);
    if (scale != 1.0 || mean != 0.0) {
      cv::Mat k;
      j.convertTo(k, type, scale, mean);
      j = k;
    }
    data += (width * height);
  }
  cv::Mat image;
  cv::merge(channels, image);
  if (depth == 1) {
    cvtColor(image, image, CV_GRAY2RGB);
  }
  return image;
}

cv::Mat to_image(const Tensor<CPUContext> &tensor, int index, float scale,
                 float mean) {
  if (tensor.IsType<float>()) {
    return to_image<float>(tensor, index, scale, mean, CV_32F);
  }
  if (tensor.IsType<uchar>()) {
    return to_image<uchar>(tensor, index, scale, mean, CV_8UC1);
  }
  LOG(FATAL) << "tensor to image for type " << tensor.meta().name()
             << " not implemented";
}

void TensorUtil::ShowImage(const std::string &title, int index, float scale,
                           float mean, bool flush) {
  auto image = to_image(tensor_, index, scale, mean);
  cvplot::Window::current().view(title).drawImage(&image);
  cvplot::Window::current().view(title).finish();
  if (flush) {
    cvplot::Window::current().view(title).flush();
  }
}

void TensorUtil::ShowImages(const std::string &name, float scale, float mean,
                            bool flush) {
  for (auto i = 0, e = (int)tensor_.dim(0); i != e; i++) {
    ShowImage(name + "-" + std::to_string(i), i, scale, mean,
              flush && (i + 1 == e));
  }
}

void TensorUtil::WriteImages(const std::string &name, float mean, bool lossy,
                             int index) {
  auto count = tensor_.dim(0);
  for (int i = 0; i < count; i++) {
    auto suffix = index >= 0 ? "_" + std::to_string(i + index) : "";
    WriteImage(name + suffix, i, mean, lossy);
  }
}

void TensorUtil::WriteImage(const std::string &name, int index, float mean,
                            bool lossy) {
  auto image = to_image(tensor_, index, 1.0, mean);
  auto filename = name + (lossy ? ".jpg" : ".png");
  vector<int> params({CV_IMWRITE_JPEG_QUALITY, 90});
  CAFFE_ENFORCE(cv::imwrite(filename, image, params),
                "unable to write to " + filename);
}

TensorCPU TensorUtil::ScaleImageTensor(int width, int height) {
  auto count = tensor_.dim(0), dim_c = tensor_.dim(1), dim_h = tensor_.dim(2),
       dim_w = tensor_.dim(3);
  std::vector<float> output;
  output.reserve(count * dim_c * height * width);
  auto input = tensor_.data<float>();
  vector<cv::Mat> channels(dim_c);
  for (int i = 0; i < count; i++) {
    for (auto &j : channels) {
      j = cv::Mat(dim_h, dim_w, CV_32F, (void *)input);
      input += (dim_w * dim_h);
    }
    cv::Mat image;
    cv::merge(channels, image);
    // image.convertTo(image, CV_8UC3, 1.0, mean);

    cv::resize(image, image, cv::Size(width, height));

    // image.convertTo(image, CV_32FC3, 1.0, -mean);
    cv::split(image, channels);
    for (auto &c : channels) {
      output.insert(output.end(), (float *)c.datastart, (float *)c.dataend);
    }
  }
  std::vector<TIndex> dims({count, dim_c, height, width});
  return TensorCPU(dims, output, NULL);
}

template <typename T>
void image_to_tensor(TensorCPU &tensor, cv::Mat &image, float mean = 128) {
  std::vector<T> data;
  image.convertTo(image, CV_32FC3, 1.0, -mean);
  vector<cv::Mat> channels(3);
  cv::split(image, channels);
  for (auto &c : channels) {
    data.insert(data.end(), (T *)c.datastart, (T *)c.dataend);
  }
  std::vector<TIndex> dims({1, 3, image.rows, image.cols});
  TensorCPU t(dims, data, NULL);
  tensor.ResizeLike(t);
  tensor.ShareData(t);
}

template <typename T>
void read_image_tensor(TensorCPU &tensor,
                       const std::vector<std::string> &filenames, int width,
                       int height, std::vector<int> &indices, float mean,
                       TensorProto::DataType type) {
  std::vector<T> data;
  data.reserve(filenames.size() * 3 * width * height);
  auto count = 0;

  for (auto &filename : filenames) {
    // load image
    auto image = cv::imread(filename);  // CV_8UC3 uchar
    // std::cout << "image size: " << image.size() << std::endl;

    if (!image.cols || !image.rows) {
      count++;
      continue;
    }

    if (image.cols != width || image.rows != height) {
      // scale image to fit
      cv::Size scaled(std::max(height * image.cols / image.rows, width),
                      std::max(height, width * image.rows / image.cols));
      cv::resize(image, image, scaled);
      // std::cout << "scaled size: " << image.size() << std::endl;

      // crop image to fit
      cv::Rect crop((image.cols - width) / 2, (image.rows - height) / 2, width,
                    height);
      image = image(crop);
      // std::cout << "cropped size: " << image.size() << std::endl;
    }

    switch (type) {
      case TensorProto_DataType_FLOAT:
        image.convertTo(image, CV_32FC3, 1.0, -mean);
        break;
      case TensorProto_DataType_INT8:
        image.convertTo(image, CV_8SC3, 1.0, -mean);
        break;
      default:
        break;
    }
    // std::cout << "value range: (" << *std::min_element((T *)image.datastart,
    // (T *)image.dataend) << ", " << *std::max_element((T *)image.datastart, (T
    // *)image.dataend) << ")" << std::endl;

    CAFFE_ENFORCE_EQ(image.channels(), 3);
    CAFFE_ENFORCE_EQ(image.rows, height);
    CAFFE_ENFORCE_EQ(image.cols, width);

    // convert NHWC to NCHW
    vector<cv::Mat> channels(3);
    cv::split(image, channels);
    for (auto &c : channels) {
      data.insert(data.end(), (T *)c.datastart, (T *)c.dataend);
    }

    indices.push_back(count++);
  }

  // create tensor
  std::vector<TIndex> dims({(TIndex)indices.size(), 3, height, width});
  TensorCPU t(dims, data, NULL);
  tensor.ResizeLike(t);
  tensor.ShareData(t);
}

void TensorUtil::ReadImages(const std::vector<std::string> &filenames,
                            int width, int height, std::vector<int> &indices,
                            float mean, TensorProto::DataType type) {
  switch (type) {
    case TensorProto_DataType_FLOAT:
      read_image_tensor<float>(tensor_, filenames, width, height, indices, mean,
                               type);
      break;
    case TensorProto_DataType_INT8:
      read_image_tensor<int8_t>(tensor_, filenames, width, height, indices,
                                mean, type);
      break;
    case TensorProto_DataType_UINT8:
      read_image_tensor<uint8_t>(tensor_, filenames, width, height, indices,
                                 mean, type);
      break;
    default:
      LOG(FATAL) << "datatype " << type << " not implemented" << std::endl;
  }
}

void TensorUtil::ReadImage(const std::string &filename, int width, int height) {
  std::vector<int> indices;
  ReadImages({filename}, width, height, indices);
}

template <typename T, typename C>
void tensor_print_type(const Tensor<C> &tensor, const std::string &name,
                       int max) {
  const auto &data = tensor.template data<T>();
  if (name.length() > 0) std::cout << name << "(" << tensor.dims() << "): ";
  for (auto i = 0; i < (tensor.size() > max ? max : tensor.size()); ++i) {
    std::cout << (float)data[i] << ' ';
  }
  if (tensor.size() > max) {
    std::cout << "... (" << *std::min_element(data, data + tensor.size()) << ","
              << *std::max_element(data, data + tensor.size()) << ")";
  }
  if (name.length() > 0) std::cout << std::endl;
}

void TensorUtil::Print(const std::string &name, int max) {
  if (tensor_.template IsType<float>()) {
    return tensor_print_type<float>(tensor_, name, max);
  }
  if (tensor_.template IsType<int>()) {
    return tensor_print_type<int>(tensor_, name, max);
  }
  if (tensor_.template IsType<uint8_t>()) {
    return tensor_print_type<uint8_t>(tensor_, name, max);
  }
  if (tensor_.template IsType<int8_t>()) {
    return tensor_print_type<int8_t>(tensor_, name, max);
  }
  std::cout << name << "?" << std::endl;
}

}  // namespace caffe2
