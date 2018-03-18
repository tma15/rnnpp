#include <iostream>

#include "expr.h"
#include "node.h"

namespace rnnpp {

const Tensor& Expression::forward() {
  g_->outputs.resize(g_->nodes().size());

//  std::cout << "forward!" << std::endl;
  for (int i=0; i <= id_; ++i) {
    Node* node = g_->nodes()[i];
//    std::cout << "node:" << i << " " << node->type() << std::endl;

    std::vector<Tensor> inputs(node->args.size());
    for (int j=0; j < node->args.size(); ++j) {
      inputs[j] = g_->outputs[node->args[j]];
    }
    node->forward(inputs, g_->outputs[i]);
  }
  return g_->outputs[id_];
}

void Expression::backward() {
  int num_nodes = g_->nodes().size();
  g_->grads.resize(num_nodes);
//  std::cout << "backward node size:" << num_nodes << std::endl;

  for (int i=0; i < num_nodes; ++i) {
    int k = g_->outputs[i].dim.size();
    g_->grads[i].dim = g_->outputs[i].dim;
    g_->grads[i].data = new float[k]; 
  }

  for (int i=0; i < g_->grads.back().dim.size(); ++i) {
    g_->grads.back().data[i] = 1.;
  }

  for (int i=num_nodes-1; i >= 0; --i) {
    Node* node = g_->nodes()[i];
    Tensor output = g_->outputs[i];
    Tensor dEdy = g_->grads[i];
//    std::cout << i << " " << node->type() << std::endl;

    std::vector<Tensor> inputs(node->args.size());
    for (int j=0; j < node->args.size(); ++j) {
      inputs[j] = g_->outputs[node->args[j]];
    }

    for (int j=0; j < node->args.size(); ++j) {
      Tensor &dEdx = g_->grads[node->args[j]];
      // dEdx = dEdy * dydx
      node->backward(inputs, output, dEdy, j, dEdx);
//      std::cout << "Expr::backward:" << std::endl;
//      std::cout << dEdx << std::endl;
    }

  }

  for (int i=0; i < g_->parameter_nodes().size(); ++i) {
//    std::cout << "i:" << g->parameter_nodes()[i] << std::endl;
    int nid = g_->parameter_nodes()[i];
//    std::cout << "parameter node in backward i:" << g_->grads[nid].dim << std::endl;
//    std::cout << g_->grads[nid] << std::endl;
    ParameterNode* n = static_cast<ParameterNode*>(g_->nodes()[nid]);
//    std::cout << n->get_param()->grad_.dim << std::endl;
    n->add_gradient(g_->grads[nid]);
  }
}

float as_scalar(const Tensor &t) {
  return t.cdata()[0];
}

Expression operator+(const Expression &a, const Expression &b) {
  int i = a.g_->nodes().size();
  Node* node = new Add({a.id(), b.id()});
  a.g_->add_node(node);
  Expression e(a.g_, i);
  return e;
}

Expression operator*(const Expression &a, const Expression &b) {
  int i = a.g_->nodes().size();
  Node* node = new Mult({a.id(), b.id()});
  a.g_->add_node(node);
  Expression e(a.g_, i);
  return e;
}

} // namespace rnnpp
