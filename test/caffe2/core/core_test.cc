#include "caffe2/util/net.h"
#include <caffe2/core/init.h>
#include <gtest/gtest.h>

namespace caffe2 {

TEST(AffineChannelTest, TestName) {
  // NetDef net;
  // auto op = net.add_op();
  // op->set_type("AffineChannel");
  // Workspace workspace;
  // EXPECT_TRUE(workspace.RunNetOnce(net));
}

}  // namespace caffe2

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  caffe2::GlobalInit(&argc, &argv);
  auto result = RUN_ALL_TESTS();
  google::protobuf::ShutdownProtobufLibrary();
  return result;
}
