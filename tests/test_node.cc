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
    };

    static Expression to_scalar(const Expression& e) {
      return sum(e);
    }
    Optimizer optimizer;


};

TEST_F(GradientTest, Add) {
  Graph g;
  Parameter p1 = optimizer.add_parameter({1, 1});
  Parameter p2 = optimizer.add_parameter({1, 1});
  Expression x = parameter(g, p1);
  Expression y = parameter(g, p2);
  Expression z = x + y;
  gradient_check(z);
}

TEST_F(GradientTest, Mult) {
  Graph g;
  Parameter p1 = optimizer.add_parameter({1, 3});
  Parameter p2 = optimizer.add_parameter({3, 1});
  Expression x = parameter(g, p1);
  Expression y = parameter(g, p2);
  Expression z = x * y;
  gradient_check(z);
}

TEST_F(GradientTest, SquaredDistance) {
  Graph g;
  Parameter p1 = optimizer.add_parameter({1, 1});
  Parameter p2 = optimizer.add_parameter({1, 1});
  Expression x = parameter(g, p1);
  Expression y = parameter(g, p2);
  Expression z = squared_distance(x, y);
  gradient_check(z);
}

TEST_F(GradientTest, Tanh) {
  Graph g;
  Parameter p1 = optimizer.add_parameter({1, 1});
  Parameter p2 = optimizer.add_parameter({1, 1});
  Expression x = parameter(g, p1);
  Expression z = tanh(x);
  gradient_check(z);
}

TEST_F(GradientTest, Sigmoid) {
  Graph g;
  Parameter p1 = optimizer.add_parameter({1, 1});
  Parameter p2 = optimizer.add_parameter({1, 1});
  Expression x = parameter(g, p1);
  Expression z = sigmoid(x);
  gradient_check(z);
}

