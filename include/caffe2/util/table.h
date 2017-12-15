#ifndef UTIL_TABLE_H
#define UTIL_TABLE_H

#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>

namespace caffe2 {

class Table {
  enum class Alignment { None, Left, Right, Internal };
  enum class Floatpoint { None, Fixed, Scientific, Hex, Default };
  struct Entry {
    std::string string;
    int width;
    Alignment align;
    Floatpoint floatpoint;
    int precision;
    char fill;
    Entry()
        : string(""),
          width(0),
          align(Alignment::None),
          floatpoint(Floatpoint::None),
          precision(0),
          fill('\0') {}
  };

 public:
  Table() {}

  void Add(const std::string &key, const Entry &entry) {
    entries_[key] = entry;
    keys_.push_back(key);
  }

  void AddFixed(const std::string &key, int width, int precision = -1) {
    Entry entry;
    entry.width = width;
    entry.floatpoint = Floatpoint::Fixed;
    entry.precision = precision + 1;
    Add(key, entry);
  }

  void AddScientific(const std::string &key, int width, int precision = -1) {
    Entry entry;
    entry.width = width;
    entry.floatpoint = Floatpoint::Scientific;
    entry.precision = precision + 1;
    Add(key, entry);
  }

  void Add(const std::string &key, int width) {
    Entry entry;
    entry.width = width;
    Add(key, entry);
  }

  std::ostream &Setup(const Entry &entry, std::ostream &stream) {
    if (entry.width > 0) {
      stream << std::setw(entry.width);
    }
    switch (entry.align) {
      case Alignment::Left:
        stream << std::left;
        break;
      case Alignment::Right:
        stream << std::right;
        break;
      case Alignment::Internal:
        stream << std::internal;
        break;
      case Alignment::None:
        break;
    }
    if (entry.fill != '\0') {
      stream << std::setfill(entry.fill);
    }
    switch (entry.floatpoint) {
      case Floatpoint::Fixed:
        stream << std::fixed;
        break;
      case Floatpoint::Scientific:
        stream << std::scientific;
        break;
      case Floatpoint::Hex:
        stream << std::hexfloat;
        break;
      case Floatpoint::Default:
        stream << std::defaultfloat;
        break;
      case Floatpoint::None:
        break;
    }
    if (entry.precision > 0) {
      stream << std::setprecision(entry.precision - 1);
    }
    return stream;
  }

  void Set(const std::string &key, float value) {
    std::stringstream stream;
    auto &entry = entries_.at(key);
    Setup(entry, stream) << value;
    entry.string = stream.str();
  }

  void WriteHeader(std::ostream &stream) const {
    for (auto &key : keys_) {
      auto &entry = entries_.at(key);
      stream << hsep_ << std::setw(entry.width) << std::setfill(' ') << key;
    }
    stream << hsep_ << std::endl;
    if (vsep_.size() > 0) {
      for (auto &key : keys_) {
        auto &entry = entries_.at(key);
        stream << hsep_ << std::setw(entry.width) << std::setfill(vsep_[0])
               << vsep_;
      }
      stream << hsep_ << std::endl;
    }
  }

  void Write(std::ostream &stream) const {
    for (auto &key : keys_) {
      auto entry = entries_.at(key);
      stream << hsep_ << entry.string;
    }
    stream << hsep_;
  }

  void Border(bool show = true) {
    if (show) {
      hsep_ = " | ";
      vsep_ = "-";
    } else {
      hsep_ = "";
      vsep_ = "";
    }
  }

 protected:
  std::map<std::string, Entry> entries_;
  std::vector<std::string> keys_;
  std::string hsep_;
  std::string vsep_;
};

std::ostream &operator<<(std::ostream &os, Table const &obj);

}  // namespace caffe2

#endif  // UTIL_TABLE_H
