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


class ParameterTest: public ::testing::Test {
  protected:
//    void SetUp() {
//    };

    LookupParameter lp;
};

TEST_F(ParameterTest, InitLookupParameter) {
  Graph g;
  lp = LookupParameter(Dim({3, 5}));

//  Expression e = lookup(g, lp, 0);
  Expression e = lookup(g, lp, 1);
  e.forward();
}


