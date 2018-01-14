#include "caffe2/util/progress.h"

#include <iomanip>
#include <sstream>

namespace caffe2 {

bool Progress::update(int step, float interval) {
  inc(step);
  if (!mark_has() || mark_lapse() < interval) {
    return false;
  }
  std::cerr << "\\|/-"[print_count++ % 4] << " " << string() << "  \r"
            << std::flush;
  mark();
  return true;
}

void Progress::wipe() {
  std::cerr << std::string(50, ' ') << '\r' << std::flush;
}

void Progress::summarize() {
  wipe();
  std::cout << report(true) << "  " << std::endl;
}

std::string seconds_to_string(int seconds) {
  std::ostringstream stream;
  stream << std::setfill('0');
  auto h = (int)seconds / 3600;
  if (h != 0) {
    stream << std::setw(2) << h << ":";
  }
  seconds -= h * 3600;
  auto m = (int)seconds / 60;
  stream << std::setw(2) << m << ":";
  seconds -= m * 60;
  stream << std::setw(2) << seconds;
  return stream.str();
}

std::string Progress::report(bool past) const {
  std::ostringstream stream;
  stream << std::fixed << std::setprecision(1);
  if (past) {
    stream << "#" << index;
  } else {
    stream << index << "/" << size << " " << percent() << "%";
  }
  auto s = past ? avg_speed() : smooth_speed();
  stream << " " << seconds_to_string(avg_lapse());
  if (size > 0 && !past) {
    stream << "+" << seconds_to_string(eta(s));
  }
  stream << " " << s << "/s";
  return stream.str();
}

std::ostream &operator<<(std::ostream &os, Progress const &obj) {
  return os << obj.string();
}

}  // namespace caffe2
