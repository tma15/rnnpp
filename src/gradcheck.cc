#include <cmath>
#include <iostream>
#include <math.h>

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

  float alpha = 5e-3;
  bool failed = false;

  for (int i=0; i < g->parameter_nodes().size(); ++i) {
//    std::cout << "Grad " << i << std::endl;
//    std::cout << g->grads[i] << std::endl;

    int nid = g->parameter_nodes()[i];
    int tensor_size = g->outputs[nid].dim.size();

    for (int j=0; j < tensor_size; ++j) {
      float old = g->outputs[nid].data[j];

      g->outputs[nid].data[j] += alpha;
      float e_p = as_scalar(expr.forward());
//      std::cout << "xp:" << g->outputs[nid].data[j] << std::endl;

      g->outputs[nid].data[j] -= 2. * alpha;
      float e_m = as_scalar(expr.forward());
//      std::cout << "xm:" << g->outputs[nid].data[j] << std::endl;

      g->outputs[nid].data[j] = old;

      float grad = (e_p - e_m) / (2 * alpha);
      float grad2 = g->grads[nid].data[j];

      float diff = fabs(grad - grad2);
      float m = std::max(fabs(grad), fabs(grad2));
      if (diff > 0.01 && m > 0.f) {
        diff /= m;
      }
      if (diff > 0.01 || std::isnan(diff)) {
        failed = true;
      }
//      std::cout << "calc:" << grad << " backp:" << grad2 << std::endl;
    }
  }

  if (failed) {
    std::cerr << "failed to gradient_check" << std::endl;
  }
}

} // namespace rnnpp

