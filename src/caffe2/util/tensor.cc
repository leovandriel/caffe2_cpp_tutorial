#include "caffe2/util/tensor.h"

#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

namespace caffe2 {

const auto screen_width = 1600;
const auto window_padding = 4;

template<typename T>
cv::Mat to_image(const Tensor<CPUContext>& tensor, int index, float mean, int type) {
  CHECK(tensor.ndim() == 4);
  auto count = tensor.dim(0), depth = tensor.dim(1), height = tensor.dim(2), width = tensor.dim(3);
  CHECK(index < count);
  auto data = tensor.data<T>() + (index * width * height * depth);
  vector<cv::Mat> channels(depth);
  for (auto& j: channels) {
    j = cv::Mat(height, width, type, (void*)data);
    data += (width * height);
  }
  cv::Mat image;
  cv::merge(channels, image);
  image.convertTo(image, CV_8UC3, 1.0, mean);
  return image;
}

cv::Mat to_image(const Tensor<CPUContext>& tensor, int index, float mean) {
  if (tensor.IsType<float>()) {
    return to_image<float>(tensor, index, mean, CV_32F);
  }
  if (tensor.IsType<uchar>()) {
    return to_image<uchar>(tensor, index, mean, CV_8UC1);
  }
  LOG(FATAL) << "tensor to image for type " << tensor.meta().name() << " not implemented";
}

void TensorUtil::ShowImage(int width, int height, int index, const std::string& title, int offset, int wait, float mean) {
  auto image = to_image(tensor_, index, mean);
  cv::resize(image, image, cv::Size(width, height));
  cv::namedWindow(title, cv::WINDOW_AUTOSIZE);
  auto max_cols = screen_width / (image.cols + window_padding);
  cv::moveWindow(title, (offset % max_cols) * (image.cols + window_padding), (offset / max_cols) * (image.rows + window_padding));
  cv::imshow(title, image);
  cv::waitKey(wait);
}

void TensorUtil::ShowImages(int width, int height, const std::string& name, float mean) {
  for (auto i = 0; i < tensor_.dim(0); i++) {
    ShowImage(width, height, i, name + "-" + std::to_string(i), i, 1, mean);
  }
}

void TensorUtil::WriteImages(const std::string& name, float mean) {
  auto count = tensor_.dim(0);
  for (int i = 0; i < count; i++) {
    auto image = to_image(tensor_, i, mean);
    auto filename = name + "_" + std::to_string(i) + ".jpg";
    vector<int> params({ CV_IMWRITE_JPEG_QUALITY, 90 });
    CHECK(cv::imwrite(filename, image, params));
    // vector<uchar> buffer;
    // cv::imencode(".jpg", image, buffer, params);
    // std::ofstream image_file(filename, std::ios::out | std::ios::binary);
    // if (image_file.is_open()) {
    //   image_file.write((char*)&buffer[0], buffer.size());
    //   image_file.close();
    // }
  }
}

TensorCPU TensorUtil::ScaleImageTensor(int width, int height) {
  auto count = tensor_.dim(0), dim_c = tensor_.dim(1), dim_h = tensor_.dim(2), dim_w = tensor_.dim(3);
  std::vector<float> output;
  output.reserve(count * dim_c * height * width);
  auto input = tensor_.data<float>();
  vector<cv::Mat> channels(dim_c);
  for (int i = 0; i < count; i++) {
    for (auto &j: channels) {
      j = cv::Mat(dim_h, dim_w, CV_32F, (void *)input);
      input += (dim_w * dim_h);
    }
    cv::Mat image;
    cv::merge(channels, image);
    // image.convertTo(image, CV_8UC3, 1.0, mean);

    cv::resize(image, image, cv::Size(width, height));

    // image.convertTo(image, CV_32FC3, 1.0, -mean);
    cv::split(image, channels);
    for (auto &c: channels) {
      output.insert(output.end(), (float *)c.datastart, (float *)c.dataend);
    }
  }
  std::vector<TIndex> dims({ count, dim_c, height, width });
  return TensorCPU(dims, output, NULL);
}

template <typename T>
void image_to_tensor(TensorCPU &tensor, cv::Mat &image, float mean = 128) {
  std::vector<T> data;
  image.convertTo(image, CV_32FC3, 1.0, -mean);
  vector<cv::Mat> channels(3);
  cv::split(image, channels);
  for (auto &c: channels) {
    data.insert(data.end(), (T *)c.datastart, (T *)c.dataend);
  }
  std::vector<TIndex> dims({ 1, 3, image.rows, image.cols });
  TensorCPU t(dims, data, NULL);
  tensor.ResizeLike(t);
  tensor.ShareData(t);
}

template <typename T>
void read_image_tensor(TensorCPU &tensor, const std::vector<std::string> &filenames, int size, std::vector<int> &indices, float mean, TensorProto::DataType type) {
  std::vector<T> data;
  data.reserve(filenames.size() * 3 * size * size);
  auto count = 0;

  for (auto &filename: filenames) {
    // load image
    auto image = cv::imread(filename); // CV_8UC3 uchar
    // std::cout << "image size: " << image.size() << std::endl;

    if (!image.cols || !image.rows) {
      count++;
      continue;
    }

    // scale image to fit
    cv::Size scale(std::max(size * image.cols / image.rows, size), std::max(size, size * image.rows / image.cols));
    cv::resize(image, image, scale);
    // std::cout << "scaled size: " << image.size() << std::endl;

    // crop image to fit
    cv::Rect crop((image.cols - size) / 2, (image.rows - size) / 2, size, size);
    image = image(crop);
    // std::cout << "cropped size: " << image.size() << std::endl;

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
    // std::cout << "value range: (" << *std::min_element((T *)image.datastart, (T *)image.dataend) << ", " << *std::max_element((T *)image.datastart, (T *)image.dataend) << ")" << std::endl;

    CHECK(image.channels() == 3);
    CHECK(image.rows == size);
    CHECK(image.cols == size);

    // convert NHWC to NCHW
    vector<cv::Mat> channels(3);
    cv::split(image, channels);
    for (auto &c: channels) {
      data.insert(data.end(), (T *)c.datastart, (T *)c.dataend);
    }

    indices.push_back(count++);
  }

  // create tensor
  std::vector<TIndex> dims({ (TIndex)indices.size(), 3, size, size });
  TensorCPU t(dims, data, NULL);
  tensor.ResizeLike(t);
  tensor.ShareData(t);
}

void TensorUtil::ReadImages(const std::vector<std::string> &filenames, int size, std::vector<int> &indices, float mean, TensorProto::DataType type) {
    switch (type) {
    case TensorProto_DataType_FLOAT:
      read_image_tensor<float>(tensor_, filenames, size, indices, mean, type);
      break;
    case TensorProto_DataType_INT8:
      read_image_tensor<int8_t>(tensor_, filenames, size, indices, mean, type);
      break;
    case TensorProto_DataType_UINT8:
      read_image_tensor<uint8_t>(tensor_, filenames, size, indices, mean, type);
      break;
    default:
      LOG(FATAL) << "datatype " << type << " not implemented";
    }
}

void TensorUtil::ReadImage(const std::string &filename, int size) {
  std::vector<int> indices;
  ReadImages({ filename }, size, indices);
}

template<typename T, typename C>
void tensor_print_type(const Tensor<C> &tensor, const std::string &name, int max) {
  const auto& data = tensor.template data<T>();
  if (name.length() > 0) std::cout << name << "(" << tensor.dims() << "): ";
  for (auto i = 0; i < (tensor.size() > max ? max : tensor.size()); ++i) {
    std::cout << (float)data[i] << ' ';
  }
  if (tensor.size() > max) {
    std::cout << "... (" << *std::min_element(data, data + tensor.size()) << "," << *std::max_element(data, data + tensor.size()) << ")";
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
