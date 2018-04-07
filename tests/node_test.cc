#include <iostream>

#include <gtest/gtest.h>

#include "../src/dim.h"
#include "../src/expr.h"
#include "../src/graph.h"
#include "../src/gradcheck.h"
#include "../src/node.h"
#include "../src/optimizer.h"
#include "../src/rnnpp.h"
#include "../src/tensor.h"

using namespace rnnpp;

class GradientTest: public ::testing::Test {
  protected:
    void SetUp() {
      p1 = optimizer.add_parameter({2, 3});
      p2 = optimizer.add_parameter({3, 2});

      p3 = optimizer.add_parameter({2, 3});
    };

    static Expression to_scalar(const Expression& e) {
      return sum(e, -1);
    }
    Optimizer optimizer;

    Graph g;
    Parameter p1, p2, p3;
};

TEST_F(GradientTest, Lookup) {
  int n_vocab = 3;
  int dim = 10;
  LookupParameter p1 = optimizer.add_lookup_parameter({n_vocab, dim});
  Expression x = lookup(g, p1, 0);
  Expression y = lookup(g, p1, 1);
  Expression z = to_scalar(x + y);
  EXPECT_TRUE(gradient_check(z));
}

TEST_F(GradientTest, Add) {
  Expression x = parameter(g, p1);
  Expression y = parameter(g, p3);
  Expression z = to_scalar(x + y);
  EXPECT_TRUE(gradient_check(z));
}

TEST_F(GradientTest, Mult) {
  Expression x = parameter(g, p1);
  Expression y = parameter(g, p2);
  Expression z = to_scalar(x * y);
  EXPECT_TRUE(gradient_check(z));
}

TEST_F(GradientTest, Divide) {
  Expression x = parameter(g, p1);
  Expression y = parameter(g, p3);
  Expression z = to_scalar(x / y);
  EXPECT_TRUE(gradient_check(z));
}

TEST_F(GradientTest, DivideConst) {
  Expression x = parameter(g, p1);
  Expression z = to_scalar(x / 4);
  EXPECT_TRUE(gradient_check(z));
}

TEST_F(GradientTest, DivideConst2) {
  Expression x = parameter(g, p1);
  Expression z = to_scalar(4 / x);
  EXPECT_TRUE(gradient_check(z));
}

TEST_F(GradientTest, Sum) {
  Expression x = parameter(g, p1);
  Expression z = sum(x, -1);
  EXPECT_TRUE(gradient_check(z));
}

TEST_F(GradientTest, SquaredDistance) {
  Expression x = parameter(g, p1);
  Expression y = parameter(g, p3);
  Expression z = to_scalar(squared_distance(x, y));
  EXPECT_TRUE(gradient_check(z));
}

TEST_F(GradientTest, Tanh) {
  Expression x = parameter(g, p1);
  Expression z = to_scalar(tanh(x));
  EXPECT_TRUE(gradient_check(z));
}

TEST_F(GradientTest, Sigmoid) {
  Expression x = parameter(g, p1);
  Expression z = to_scalar(sigmoid(x));
  EXPECT_TRUE(gradient_check(z));
}

