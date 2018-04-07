#include <iostream>

#include <gtest/gtest.h>

#include "../src/dim.h"
#include "../src/tensor.h"

using namespace rnnpp;


class DimTest: public ::testing::Test {
  protected:
    void SetUp() {
      d1 = Dim({2, 3, 2});
      d2 = Dim({2, 3, 2});
      d3 = Dim({3, 1, 4});
    };

    Dim d1, d2, d3;
};

TEST_F(DimTest, Shape) {
  EXPECT_EQ(d1.shape.size(), 3);
}

TEST_F(DimTest, Stride) {
  EXPECT_EQ(d1.stride[0], 6);
  EXPECT_EQ(d1.stride[1], 2);
  EXPECT_EQ(d1.stride[2], 1);
}

TEST_F(DimTest, Equal) {
  EXPECT_TRUE(d1 == d2);
  EXPECT_FALSE(d1 == d3);
}
