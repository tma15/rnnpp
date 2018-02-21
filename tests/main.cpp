#include <iostream>

#include "../src/expr.h"
#include "../src/rnnpp.h"
#include "../src/node.h"
#include "../src/graph.h"

using namespace rnnpp;

int main(int argc, char** argv) {
  Graph g;

  std::vector<float> x_val{
    1., 2.,
    3., 4.,
    5., 6.,
  };
  Expression x = input(g, Dim({2}, 3), x_val);

  std::vector<float> y_val{
    1., 2., 3.,
  };
  Expression y = input(g, Dim({1}, 3), y_val);

//  Expression w = parameter(g, {3, 2});
  Expression w = parameter(g, {1, 2});

  Expression y_pred = w * x;

  Expression loss = squared_distance(y_pred, y);

  loss.forward();

  return 0;
}
