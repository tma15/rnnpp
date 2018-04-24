#include <iostream>

#include "expr.h"
#include "node.h"

namespace rnnpp {

const Tensor& Expression::forward() {
  g_->outputs.resize(g_->nodes().size());

  for (int i=0; i <= id_; ++i) {
    Node* node = g_->nodes()[i];

    std::vector<Tensor> inputs(node->args.size());
    for (int j=0; j < node->args.size(); ++j) {
      inputs[j] = g_->outputs[node->args[j]];
    }

    node->forward(inputs, g_->outputs[i]);
  }
  return g_->outputs[id_];
}

std::vector<Tensor> Expression::forward2() {
  g_->outputs.resize(g_->n_outputs());

  for (int i=0; i <= id_; ++i) {
    Node* node = g_->nodes()[i];

    std::vector<Tensor> inputs(node->args.size());
    for (int j=0; j < node->args.size(); ++j) {
      inputs[j] = g_->outputs[node->args[j]];
    }

    std::vector<Tensor*> outputs(node->args_out.size());
    for (int j=0; j < node->args_out.size(); ++j) {
      outputs[j] = &g_->outputs[node->args_out[j]];
    }
    node->forward2(inputs, outputs);
  }
  std::vector<Tensor> ret(g_->nodes()[id_]->args_out.size());
  for (int j=0; j < g_->nodes()[id_]->args_out.size(); ++j) {
    ret[j] = g_->outputs[g_->nodes()[id_]->args_out[j]];
  }
  return ret;
}

void Expression::backward() {
  int num_nodes = g_->nodes().size();
  g_->grads.resize(num_nodes);

  for (int i=0; i < num_nodes; ++i) {
    int k = g_->outputs[i].dim.size();
    g_->grads[i].dim = g_->outputs[i].dim;
    g_->grads[i].data = new float[k]; 
  }

  g_->grads.back() = Scalar(1.);

  for (int i=num_nodes-1; i >= 0; --i) {
    Node* node = g_->nodes()[i];
    Tensor output = g_->outputs[i];
    Tensor dEdy = g_->grads[i];

    std::vector<Tensor> inputs(node->args.size());
    for (int j=0; j < node->args.size(); ++j) {
      inputs[j] = g_->outputs[node->args[j]];
    }

    for (int j=0; j < node->args.size(); ++j) {
      Tensor &dEdx = g_->grads[node->args[j]];
      node->backward(inputs, output, dEdy, j, dEdx);
    }
  }

  for (int i=0; i < g_->parameter_nodes().size(); ++i) {
    int nid = g_->parameter_nodes()[i];
    ParameterNodeBase* n = static_cast<ParameterNodeBase*>(g_->nodes()[nid]);
    n->add_gradient(g_->grads[nid]);
  }
}

void Expression::backward2() {
  int n_out = g_->n_outputs();
//  int num_nodes = g_->nodes().size();
//  g_->grads.resize(num_nodes);
  g_->grads.resize(g_->n_outputs());

  for (int i=0; i < n_out; ++i) {
    int k = g_->outputs[i].dim.size();
    g_->grads[i].dim = g_->outputs[i].dim;
    g_->grads[i].data = new float[k]; 
  }

  g_->grads.back() = Scalar(1.);

  for (int i=n_out-1; i >= 0; --i) {
    Node* node = g_->nodes()[i];
//    std::cout << i << " " << node->type() << std::endl;

//    Tensor output = g_->outputs[i];
//    Tensor dEdy = g_->grads[i];

    std::vector<Tensor> inputs(node->args.size());
    for (int j=0; j < node->args.size(); ++j) {
      inputs[j] = g_->outputs[node->args[j]];
    }

    std::vector<Tensor> outputs(node->args_out.size());
    for (int j=0; j < node->args_out.size(); ++j) {
      outputs[j] = g_->outputs[node->args_out[j]];
    }

    std::vector<Tensor> dEdy(node->args_out.size());
    for (int j=0; j < node->args_out.size(); ++j) {
      dEdy[j] = g_->grads[node->args_out[j]];
    }

    for (int j=0; j < node->args.size(); ++j) {
      Tensor &dEdx = g_->grads[node->args[j]];
      node->backward2(inputs, outputs, dEdy, j, dEdx);
    }
  }

  for (int i=0; i < g_->parameter_nodes().size(); ++i) {
    int nid = g_->parameter_nodes()[i];
    ParameterNodeBase* n = static_cast<ParameterNodeBase*>(g_->nodes()[nid]);
    n->add_gradient(g_->grads[nid]);
  }
}


float as_scalar(const Tensor &t) {
  return t.cdata()[0];
}

Expression operator+(const Expression &a, const Expression &b) {
  int i = a.g_->nodes().size();
//  Node* node = new Add({a.id(), b.id()});
  Node* node = new Add({a.id(), b.id()}, {i});
  a.g_->add_node(node);
  Expression e(a.g_, i);
  return e;
}

Expression operator*(const Expression &a, const Expression &b) {
  int i = a.g_->nodes().size();
  Node* node = new Mult({a.id(), b.id()}, {i});
  a.g_->add_node(node);
  Expression e(a.g_, i);
  return e;
}

Expression operator/(const Expression &a, const Expression &b) {
  int i = a.g_->nodes().size();
  Node* node = new Divide({a.id(), b.id()}, {i});
  a.g_->add_node(node);
  Expression e(a.g_, i);
  return e;
}

Expression operator/(const Expression &a, float b) {
  int i = a.g_->nodes().size();
  bool rhs_is_const = true;
  Node* node = new DivideConst({a.id()}, {i}, b, rhs_is_const);
  a.g_->add_node(node);
  Expression e(a.g_, i);
  return e;
}

Expression operator/(float a, const Expression &b) {
  int i = b.g_->nodes().size();
  bool rhs_is_const = false;
  Node* node = new DivideConst({b.id()}, {i}, a, rhs_is_const);
  b.g_->add_node(node);
  Expression e(b.g_, i);
  return e;
}

Expression concat(const std::initializer_list<Expression> &xs, int axis) {
  std::vector<int> ids(xs.size());
  int k = 0;
  Graph* g;
  for (auto x=xs.begin(); x != xs.end(); ++x) {
    ids[k++] = x->id();
    g = x->g_;
  }
  int nid = g->nodes().size();

  Node* node = new Concat(ids, {nid}, axis);
  g->add_node(node);
  Expression e(g, nid);
  return e;
}

std::vector<Expression> split(const Expression &x, int n, int axis) {

  std::vector<Expression> ret(n);
  std::vector<int> outids(n);
  int nid = x.g_->nodes().size();
  for (int i=0; i < n; ++i) {
    Expression e(x.g_, nid++);
    outids[i] = nid;
  }
  Node* node = new Split({x.id()}, outids, axis);
  x.g_->add_node(node);

  return ret;
//  int k = 0;
//  for (int id=x.id(); id < x.id() + n; ++id) {
//    outids[k++] = id;
//  }
//  int nid = x.g_->nodes().size();

//  Expression e(x.g_, nid);
//  return e;
}


} // namespace rnnpp
