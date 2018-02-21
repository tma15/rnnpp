#ifndef RNNPP_NODE_H_
#define RNNPP_NODE_H_

#include <initializer_list>
#include <vector>

#include "dim.h"
#include "tensor.h"

namespace rnnpp {

class Node {
  public:
    Node() {}

    Node(std::initializer_list<int> a): args(a) {}

    ~Node() {}

    virtual void forward(const std::vector<Tensor>& inputs, Tensor &output)=0;

    virtual void backward(const std::vector<Tensor>& inputs, Tensor &output){};

    std::vector<int> args;

    virtual std::string type()=0;

    Dim dim;
};


class InputNode: public Node {
  public:
    InputNode(): Node() {}

    InputNode(std::vector<float> *data): Node(), data_(data) {}

    InputNode(const Dim& d, std::vector<float> *data): Node(), data_(data) {
      dim = d;
    }

    ~InputNode() {}

    void forward(const std::vector<Tensor>& inputs, Tensor &output);

    std::string type() { return "InputNode"; }

  private:
    std::vector<float> *data_;
};


class ParameterNode: public Node {
  public:
    ParameterNode(): Node() {}

    ParameterNode(std::initializer_list<int> d): Node() {
      dim = Dim(d);
      int n = 1;
      for (auto it=d.begin(); it != d.end(); ++it) {
        n *= *it;
      }

      data_ = std::vector<float>(n);
      for (int i=0; i < n; ++i) {
        data_[i] = 1.;
      }
    }

    ~ParameterNode() {}

    void forward(const std::vector<Tensor>& inputs, Tensor &output);

    std::string type() { return "ParameterNode"; }

  private:
    std::vector<float> data_;
//    Tensor tensor;
};


class Mult: public Node {
  public:
    Mult(): Node() {}

    Mult(std::initializer_list<int> a): Node(a) {}

    ~Mult(){}
    void forward(const std::vector<Tensor>& inputs, Tensor &output);

    std::string type() { return "Mult"; }
};


class SquaredDistance: public Node {
  public:
    SquaredDistance(): Node() {}

    SquaredDistance(std::initializer_list<int> a): Node(a) {}

    ~SquaredDistance(){}

    void forward(const std::vector<Tensor>& inputs, Tensor &output);

    std::string type() { return "SquaredDistance"; }
};



class Embed: public Node {
  public:
    Embed(): Node() {}

    ~Embed() {}

    void forward(const std::vector<Tensor>& inputs, Tensor &output) {}

    std::string type() { return "Embed"; }
};

} // namespace rnnpp

#endif // RNNPP_NODE_H_
