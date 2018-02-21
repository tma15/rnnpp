#include <iostream>

#include "expr.h"
#include "node.h"

namespace rnnpp {

void Expression::forward() {
//  g_->inputs.resize(g_->nodes().size());
  g_->outputs.resize(g_->nodes().size());

  std::cout << "forward!" << std::endl;
//  for (int i=0; i < g_->nodes().size(); ++i) {
  for (int i=0; i <= id_; ++i) {
    Node* node = g_->nodes()[i];
    std::cout << "node:" << i << " " << node->type() << std::endl;

    std::vector<Tensor> inputs(node->args.size());
    for (int j=0; j < node->args.size(); ++j) {
      inputs[j] = g_->outputs[node->args[j]];
    }

    node->forward(inputs, g_->outputs[i]);
  }
}

Expression operator*(const Expression &a, const Expression &b) {
  int i = a.g_->nodes().size();
  Node* node = new Mult({a.id(), b.id()});
  a.g_->add_node(node);
  Expression e(a.g_, i);
  return e;
}

} // namespace rnnpp
