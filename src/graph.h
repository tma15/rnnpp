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

    std::vector<Tensor> outputs;

    // dE/dx = dE/df * df/dx
    //
    // Sum:
    //  y = a + b
    //  dE/da = dE/dy * dy/da
    //        = dE/dy
    //
    // Mult:
    //  y = a * b
    //  dE/da = dE/dy * dy/da
    //        = dE/dy * b
    //  dE/db = dE/dy * dy/db
    //        = dE/dy * a
    std::vector<Tensor> grads;

  private:
    std::vector<Node*> nodes_;
    std::vector<int> parameter_node_ids_;

};

} // namespace rnnpp

#endif // RNNPP_GRAPH_H_
