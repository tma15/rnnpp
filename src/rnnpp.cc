#include <initializer_list>
#include <vector>

#include "graph.h"
#include "node.h"
#include "rnnpp.h"

namespace rnnpp {

Expression input(Graph &g, std::vector<float> &value) {
  int i = g.nodes().size();
  Node* node = new InputNode(&value);
  g.add_node(node);
  Expression e(&g, i);
  return e;
}

Expression input(Graph &g, const Dim &dim, std::vector<float> &value) {
  int i = g.nodes().size();
  Node* node = new InputNode(dim, &value);
  g.add_node(node);
  Expression e(&g, i);
  return e;
}

Expression parameter(Graph &g, std::initializer_list<int> dim) {
  int i = g.nodes().size();
  Node* node = new ParameterNode(dim);
  g.add_node(node);

  Expression e(&g, i);
  return e;
}


Expression squared_distance(const Expression &a, const Expression &b) {
  int i = a.g_->nodes().size();
  Node* node = new SquaredDistance({a.id(), b.id()});
  a.g_->add_node(node);

  Expression e(a.g_, i);
  return e;
}

} // namespace rnnpp
