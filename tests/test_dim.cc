#include <iostream>

#include <gtest/gtest.h>

#include "../src/dim.h"
#include "../src/tensor.h"

using namespace rnnpp;


class DimTest: public ::testing::Test {
  protected:
    void SetUp() {
      d = Dim({2, 3, 2});
    };
    Dim d;
};

TEST_F(DimTest, DimTest1) {
  EXPECT_EQ(d.shape.size(), 3);
  EXPECT_EQ(d.stride[0], 6);
  EXPECT_EQ(d.stride[1], 2);
  EXPECT_EQ(d.stride[2], 1);
}

