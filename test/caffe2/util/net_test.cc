#include "caffe2/util/net.h"
#include <gtest/gtest.h>
#include <caffe2/core/init.h>

namespace caffe2 {

TEST(NetUtilTest, TestName) {
  NetDef model;
  NetUtil net(model, "test");
  EXPECT_EQ(model.name(), "test");
}

}  // namespace caffe2

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  caffe2::GlobalInit(&argc, &argv);
  auto result = RUN_ALL_TESTS();
  google::protobuf::ShutdownProtobufLibrary();
  return result;
}
