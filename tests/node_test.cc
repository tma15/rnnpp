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
      return sum(e, -1);
    }
    Optimizer optimizer;
};

TEST_F(GradientTest, Lookup) {
  Graph g;
  int n_vocab = 3;
  int dim = 10;
  LookupParameter p1 = optimizer.add_lookup_parameter({n_vocab, dim});
  Expression x = lookup(g, p1, 0);
  Expression y = lookup(g, p1, 1);
  Expression z = to_scalar(x + y);
  gradient_check(z);
}


TEST_F(GradientTest, Add) {
  Graph g;
  Parameter p1 = optimizer.add_parameter({2, 3});
  Parameter p2 = optimizer.add_parameter({3, 2});
  Expression x = parameter(g, p1);
  Expression y = parameter(g, p2);
  Expression z = to_scalar(x + y);
  gradient_check(z);
}

TEST_F(GradientTest, Mult) {
  Graph g;
  Parameter p1 = optimizer.add_parameter({2, 3});
  Parameter p2 = optimizer.add_parameter({3, 2});
  Expression x = parameter(g, p1);
  Expression y = parameter(g, p2);
  Expression z = to_scalar(x * y);
  gradient_check(z);
}

TEST_F(GradientTest, Sum) {
  Graph g;
  Parameter p1 = optimizer.add_parameter({2, 2});
  Expression x = parameter(g, p1);
  Expression z = sum(x, -1);
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
  Parameter p1 = optimizer.add_parameter({3, 4});
  Expression x = parameter(g, p1);
  Expression z = to_scalar(tanh(x));
  gradient_check(z);
}

TEST_F(GradientTest, Sigmoid) {
  Graph g;
  Parameter p1 = optimizer.add_parameter({3, 4});
  Expression x = parameter(g, p1);
  Expression z = to_scalar(sigmoid(x));
  gradient_check(z);
}

