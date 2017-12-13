#include "caffe2/util/progress.h"

#include <iomanip>
#include <sstream>

namespace caffe2 {

void Progress::update(int step, float interval) {
  inc(step);
  if (mark_has() && mark_lapse() > interval) {
    std::cerr << "\\|/-"[print_count++ % 4] << " " << string() << "  \r"
              << std::flush;
    mark();
  }
}

void Progress::wipe() {
  std::cerr << std::string(50, ' ') << '\r' << std::flush;
}

void Progress::summarize() {
  wipe();
  std::cout << report(true) << "  " << std::endl;
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
  if (size > 0) {
    auto e = past ? avg_lapse() : eta(s);
    auto h = (int)e / 3600;
    e -= h * 3600;
    auto m = (int)e / 60;
    e -= m * 60;
    auto s = (int)e;
    stream << " " << std::setfill('0') << std::setw(2) << h << ":"
           << std::setw(2) << m << ":" << std::setw(2) << s;
  }
  stream << " " << s << "/s";
  return stream.str();
}

std::ostream &operator<<(std::ostream &os, Progress const &obj) {
  return os << obj.string();
}

}  // namespace caffe2
