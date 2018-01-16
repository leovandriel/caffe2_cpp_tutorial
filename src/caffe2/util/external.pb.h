#ifndef MODEL_EXTERNAL_H
#define MODEL_EXTERNAL_H

// HACK: on

#include <caffe2/core/net.h>

namespace protobuf_external_2eproto {
  void InitDefaultsNetDef();
  void AddDescriptors();
};

namespace caffe2 {
  void protobuf_AddDesc_external_2eproto();
}

// HACK: off

#endif  // MODEL_EXTERNAL_H
