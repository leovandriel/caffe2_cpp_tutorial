#include <caffe2/core/init.h>
#include <gtest/gtest.h>
#include "caffe2/util/net.h"

namespace caffe2 {

TEST(NetUtilTest, TestGradientBasic) {
  NetDef def;
  NetUtil net(def);
  net.AddFcOp("in", "w", "b", "output");
  net.AddGradientOps();
  auto op = def.op(def.op_size() - 1);
  EXPECT_EQ(op.type(), "FCGradient");
  EXPECT_EQ(op.input(0), "in");
  EXPECT_EQ(op.input(1), "w");
  EXPECT_EQ(op.input(2), "output_grad");
}

TEST(NetUtilTest, TestGradientStop) {
  NetDef def;
  NetUtil net(def);
  net.AddFcOp("in", "w", "b", "t1");
  net.AddFcOp("t1", "w", "b", "t2");
  net.AddStopGradientOp("t2");
  net.AddFcOp("t2", "w", "b", "output");
  net.AddGradientOps();
  EXPECT_EQ(def.op_size(), 5);
  auto op = def.op(def.op_size() - 1);
  EXPECT_EQ(op.type(), "FCGradient");
  EXPECT_EQ(op.input(0), "t2");
}

}  // namespace caffe2

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  caffe2::GlobalInit(&argc, &argv);
  auto result = RUN_ALL_TESTS();
  google::protobuf::ShutdownProtobufLibrary();
  return result;
}
