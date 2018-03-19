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

  std::vector<float> x_val{
    1., 2.,
//    3., 4.,
//    5., 6.,
  };
  Expression x = input(g, Dim({2, 1}, 1), x_val);
//  Expression x = input(g, Dim({2, 1}, 3), x_val);

  std::vector<float> y_val{
    1.,
//    2.,
//    3.,
  };
  Expression y = input(g, Dim({1, 1}, 1), y_val);
//  Expression y = input(g, Dim({1, 1}, 3), y_val);

  int n_hidden = 4;
  Parameter p_w = optimizer.add_parameter({n_hidden, 2});
  Parameter p_b = optimizer.add_parameter({n_hidden, 1});
  Parameter p_w2 = optimizer.add_parameter({1, n_hidden});
  Parameter p_b2 = optimizer.add_parameter({1, 1});

  Expression w = parameter(g, p_w);
  Expression b = parameter(g, p_b);
  Expression w2 = parameter(g, p_w2);
  Expression b2 = parameter(g, p_b2);

//  Expression h = tanh(w * x + b);
  Expression h = sigmoid(w * x + b);
  Expression y_pred = w2 * h + b2;

  Expression loss = squared_distance(y_pred, y);

  for (int i=0; i < 1000; ++i) {
    float err = 0.;
    for (unsigned mi = 0; mi < 4; ++mi) {
      bool x1 = mi % 2;
      bool x2 = (mi / 2) % 2;
      x_val[0] = x1 ? 1 : -1;
      x_val[1] = x2 ? 1 : -1;
      y_val[0] = (x1 != x2) ? 1 : -1;

      std::cout << "x:" << x_val[0] << ", " << x_val[1] 
                << " y:" << y_val[0] << std::endl;

      float err_i = as_scalar(loss.forward());
      err += err_i;
//      std::cout << "true_y:" << y_val[0] << std::endl;
//      std::cout << "E_i: " << err_i << std::endl;
      loss.backward();
      optimizer.update();
//      std::cout << "===" << std::endl;

    }
    std::cout << "E:" << err/4 << std::endl;
  }

  return 0;
}
