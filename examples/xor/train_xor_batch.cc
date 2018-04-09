#include <iostream>

#include "../src/expr.h"
#include "../src/rnnpp.h"
#include "../src/node.h"
#include "../src/optimizer.h"
#include "../src/graph.h"

using namespace rnnpp;

int main(int argc, char** argv) {
  Graph g;
  Optimizer optimizer;

  int n_batch = 4;

  std::vector<float> x_val{
    1., 1.,
    1., -1.,
    -1., 1.,
    -1., -1.,
  };
  Expression x = input(g, Dim({2, 1}, n_batch), x_val);

  std::vector<float> y_val{
    -1., 1., 1., -1.,
  };
  Expression y = input(g, Dim({1, 1}, n_batch), y_val);

  int n_hidden = 8;
  Parameter p_w = optimizer.add_parameter({n_hidden, 2});
  Parameter p_b = optimizer.add_parameter({n_hidden, 1});
  Parameter p_w2 = optimizer.add_parameter({1, n_hidden});
  Parameter p_b2 = optimizer.add_parameter({1, 1});

  Expression w = parameter(g, p_w);
  Expression b = parameter(g, p_b);
  Expression w2 = parameter(g, p_w2);
  Expression b2 = parameter(g, p_b2);

  Expression h = tanh(w * x + b);
  Expression y_pred = w2 * h + b2;

  Expression losses = squared_distance(y_pred, y);

  int axis = 2;
  Expression loss = sum(losses, axis)/n_batch;

  for (int i=0; i < 30; ++i) {
    float err = as_scalar(loss.forward());
    loss.backward();
    optimizer.update();
    std::cout << "E:" << err << std::endl;
  }

  return 0;
}
