#include <iostream>

#include <gtest/gtest.h>

#include "../src/dim.h"
#include "../src/expr.h"
#include "../src/graph.h"
#include "../src/optimizer.h"
#include "../src/rnnpp.h"
#include "../src/tensor.h"

using namespace rnnpp;


class ExprTest: public ::testing::Test {
  protected:
    void SetUp() {
      p1 = optimizer.add_parameter({2, 3});
      p2 = optimizer.add_parameter({2, 3});
    };

    Expression expr;
    Graph g;
    Optimizer optimizer;
    Parameter p1, p2;
};


TEST_F(ExprTest, Forward) {
  Expression x = parameter(g, p1);
  Expression y = parameter(g, p2);
  Expression z = x + y;
  z.forward2();
}


