#include "caffe2/util/table.h"

#include <iomanip>
#include <sstream>

namespace caffe2 {

std::ostream &operator<<(std::ostream &os, Table const &obj) {
  obj.Write(os);
  return os;
}

}  // namespace caffe2
