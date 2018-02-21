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

    void add_node(Node* node) { nodes_.push_back(node); }

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
    std::vector<Tensor> grads;

  private:
    std::vector<Node*> nodes_;

};

} // namespace rnnpp

#endif // RNNPP_GRAPH_H_
