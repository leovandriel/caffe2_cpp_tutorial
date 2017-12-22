#include "cvplot/color.h"

#include <cmath>

namespace cvplot {

Color Color::alpha(uint8_t alpha) const { return Color(r, g, b, alpha); }

Color Color::gray(uint8_t v) { return Color(v, v, v); }

Color Color::index(uint32_t index, uint32_t density, float avoid,
                   float range) {  // avoid greens by default
  if (avoid > 0) {
    auto step = density / (6 - range);
    auto offset = (avoid + range / 2) * step;
    index = offset + index % density;
    density += step * range;
  }
  auto hue = index % density * 6.f / density;
  return Color::cos(hue);
}

Color Color::hash(const std::string &seed) {
  return Color::index(std::hash<std::string>{}(seed));
}

Color Color::hue(float hue) {
  Color color;
  auto i = (int)hue;
  auto f = (hue - i) * (256 - paleness * 2) + paleness;
  switch (i) {
    case 0:
      color.r = 256 - paleness;
      color.g = f;
      color.b = paleness;
      break;
    case 1:
      color.r = 256 - f;
      color.g = 256 - paleness;
      color.b = paleness;
      break;
    case 2:
      color.r = paleness;
      color.g = 256 - paleness;
      color.b = f;
      break;
    case 3:
      color.r = paleness;
      color.g = 256 - f;
      color.b = 256 - paleness;
      break;
    case 4:
      color.r = f;
      color.g = paleness;
      color.b = 256 - paleness;
      break;
    case 5:
    default:
      color.r = 256 - paleness;
      color.g = paleness;
      color.b = 256 - f;
      break;
  }
  return color;
}

Color Color::cos(float hue) {
  return Color((std::cos(hue * 1.047) + 1) * 127.9,
               (std::cos((hue - 2) * 1.047) + 1) * 127.9,
               (std::cos((hue - 4) * 1.047) + 1) * 127.9);
}

}  // namespace cvplot
