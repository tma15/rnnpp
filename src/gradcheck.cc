#include <iostream>

#include "gradcheck.h"
#include "node.h"
#include "rnnpp.h"

namespace rnnpp {

void gradient_check(Expression &expr) {
//  const std::vector<Node*> &nodes = expr.g_->nodes();
//  std::cout << "gradient_check:" << std::endl;

  expr.forward();
  expr.backward();

  Graph* g = expr.g_;

//  std::vector<int> args = g->node(expr.id())->args;
//  for (int i=0; i < args.size(); ++i) {
//    std::cout << "inputs " << args[i] << ":" << std::endl;
//    std::cout << g->outputs[args[i]] << std::endl;
//  }
//  std::cout << "output:" << std::endl;
//  std::cout << g->outputs[expr.id()] << std::endl;

  float alpha = 5e-3;

  for (int i=0; i < g->parameter_nodes().size(); ++i) {
    std::cout << "Grad " << i << std::endl;
    std::cout << g->grads[i] << std::endl;

    int nid = g->parameter_nodes()[i];
    int tensor_size = g->outputs[nid].dim.size();

    for (int j=0; j < tensor_size; ++j) {
      float old = g->outputs[nid].data[j];

      g->outputs[nid].data[j] += alpha;
      float e_p = as_scalar(expr.forward());

      g->outputs[nid].data[j] -= 2. * alpha;
      float e_m = as_scalar(expr.forward());

      g->outputs[nid].data[j] = old;

      float grad = (e_p - e_m) / (2 * alpha);
      float grad2 = g->grads[nid].data[j];
//      std::cout << "e_p:" << e_p << " e_m:" << e_m << std::endl;
      std::cout << "calc:" << grad << " backp:" << grad2 << std::endl;
//      std::cout << std::endl;
    }

  }
}

} // namespace rnnpp

