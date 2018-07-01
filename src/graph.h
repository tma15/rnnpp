#ifndef RNNPP_GRAPH_H_
#define RNNPP_GRAPH_H_

#include "node.h"
#include "tensor.h"

namespace rnnpp {

class Graph {
  public:
    Graph(){}
    ~Graph(){}

    const std::vector<Node*>& nodes() { return nodes_; }

    Node* node(int i) { return nodes_[i]; }

    const std::vector<int>& parameter_nodes() { return parameter_node_ids_; }

    void add_node(Node* node) { nodes_.push_back(node); }
    void add_parameter_node(int i) { parameter_node_ids_.push_back(i); }

    int n_outputs() {
      int n_out = 0;
      for (int i=0; i < nodes().size(); ++i) {
        n_out += nodes()[i]->n_out();
      }
      return n_out;
    }

    std::vector<Tensor> outputs;

    std::vector<Tensor> grads;

  private:
    std::vector<Node*> nodes_;
    std::vector<int> parameter_node_ids_;

};

} // namespace rnnpp

#endif // RNNPP_GRAPH_H_
