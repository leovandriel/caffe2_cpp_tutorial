#ifndef UTIL_PROGRESS_H
#define UTIL_PROGRESS_H

#include <iostream>

namespace caffe2 {

const auto mark_size = 60;

class Progress {
 public:
  Progress(size_t size = 0) : size(size) { reset(); }
  void reset() {
    index = 0;
    start_time = clock();
    for (auto i = 0; i < mark_size; i++) {
      mark_index[i] = index;
      mark_time[i] = start_time;
    }
    mark_count = 0;
  }
  void inc(size_t step = 1) { index += step; }
  void set(size_t index_) { index = index_; }
  void mark() {
    mark_index[mark_count % mark_size] = index;
    mark_time[mark_count % mark_size] = clock();
    mark_count++;
  }
  float mark_lapse() const {
    auto i = (mark_count + (mark_size - 1)) % mark_size;
    return (float)(clock() - mark_time[i]) / CLOCKS_PER_SEC;
  }
  float smooth_lapse() const {
    return (float)(clock() - mark_time[mark_count % mark_size]) /
           CLOCKS_PER_SEC;
  }
  float avg_lapse() const {
    return (float)(clock() - start_time) / CLOCKS_PER_SEC;
  }
  float mark_speed() const {
    auto i = (mark_count + (mark_size - 1)) % mark_size;
    return (index - mark_index[i]) / mark_lapse();
  }
  float smooth_speed() const {
    return (index - mark_index[mark_count % mark_size]) / smooth_lapse();
  }
  float avg_speed() const { return index / avg_lapse(); }
  float mark_has() const {
    return index > mark_index[(mark_count + (mark_size - 1)) % mark_size];
  }
  float smooth_has() const {
    return index > mark_index[mark_count % mark_size];
  }
  float avg_has() const { return index > 0; }
  float percent() const { return size > 0 ? 100.f * index / size : -1.f; }
  float eta(float speed) const {
    return speed > 0 && size > 0 ? (size - index) / speed : -1.f;
  }
  bool update(int step = 1, float interval = 1);
  void wipe();
  void summarize();
  std::string report(bool past = false) const;
  std::string string() const { return report(index >= size); }

 protected:
  size_t index;
  size_t size;
  clock_t start_time;
  int mark_count;
  size_t mark_index[mark_size];
  clock_t mark_time[mark_size];
  size_t print_count;
};

std::ostream &operator<<(std::ostream &os, Progress const &obj);

}  // namespace caffe2

#endif  // UTIL_PROGRESS_H
