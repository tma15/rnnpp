#ifndef RNNPP_NODE_H_
#define RNNPP_NODE_H_

#include <initializer_list>
#include <vector>

#include "dim.h"
#include "parameter.h"
#include "tensor.h"

namespace rnnpp {

class Node {
  public:
    Node() {}

    Node(std::initializer_list<int> a): args(a) {}

    ~Node() {}

    virtual void forward(const std::vector<Tensor>& inputs, Tensor &output)=0;

    virtual void backward(const std::vector<Tensor>& inputs, const Tensor &output,
        const Tensor &dEdy, int ii, Tensor &dEdxi)=0;

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

    void backward(const std::vector<Tensor>& inputs, const Tensor &output,
        const Tensor &dEdy, int ii, Tensor &dEdxi){};

    std::string type() { return "InputNode"; }

  private:
    std::vector<float> *data_;
};


class ParameterNode: public Node {
  public:
    ParameterNode(): Node() {}

    ParameterNode(Parameter p): Node() {
      param = p;
      dim = p.value.dim;
    }

    ~ParameterNode() {}

    void forward(const std::vector<Tensor>& inputs, Tensor &output);

    void backward(const std::vector<Tensor>& inputs, const Tensor &output,
        const Tensor &dEdy, int ii, Tensor &dEdxi){};

    Parameter* get_param() { return &param; }

    void add_gradient(const Tensor &dEdy) {
      param.grad += dEdy;
    }

    std::string type() { return "ParameterNode"; }

  private:
    Parameter param;
};

//class LookupNode: public ParameterNode {
//  public:
//    LookupNode(): ParameterNode() {}

//    LookupNode(LookupParameter p): ParameterNode() {
//      param = p;
//      dim = p.data_.dim;
//    }
//};

/**
 * y = x^2
 * dEdx = dEdy * dydx
 *      = dEdy * 2 * x
 */
class Square: public Node {
  public:
    Square(): Node() {}

    Square(std::initializer_list<int> a): Node(a) {}

    ~Square(){}

    void forward(const std::vector<Tensor>& inputs, Tensor &output);

    void backward(const std::vector<Tensor>& inputs, const Tensor &output,
        const Tensor &dEdy, int ii, Tensor &dEdxi);

    std::string type() { return "Square"; }
};

/**
 * y = a + b + c
 * dEda = dEdy * dyda = dEdy
 * dEdb = dEdy * dydb = dEdy
 * dEdc = dEdy * dydc = dEdy
 */
class Sum: public Node {
  public:
    Sum(): Node() {}

    Sum(std::initializer_list<int> a): Node(a) {}

    ~Sum(){}

    void forward(const std::vector<Tensor>& inputs, Tensor &output);

    void backward(const std::vector<Tensor>& inputs, const Tensor &output,
        const Tensor &dEdy, int ii, Tensor &dEdxi);

    std::string type() { return "Sum"; }
};


class Add: public Node {
  public:
    Add(): Node() {}

    Add(std::initializer_list<int> a): Node(a) {}

    ~Add(){}

    void forward(const std::vector<Tensor>& inputs, Tensor &output);

    void backward(const std::vector<Tensor>& inputs, const Tensor &output,
        const Tensor &dEdy, int ii, Tensor &dEdxi);

    std::string type() { return "Add"; }
};


class Mult: public Node {
  public:
    Mult(): Node() {}

    Mult(std::initializer_list<int> a): Node(a) {}

    ~Mult(){}

    void forward(const std::vector<Tensor>& inputs, Tensor &output);

    void backward(const std::vector<Tensor>& inputs, const Tensor &output,
        const Tensor &dEdy, int ii, Tensor &dEdxi);

    std::string type() { return "Mult"; }
};


class SquaredDistance: public Node {
  public:
    SquaredDistance(): Node() {}

    SquaredDistance(std::initializer_list<int> a): Node(a) {}

    ~SquaredDistance(){}

    void forward(const std::vector<Tensor>& inputs, Tensor &output);

    void backward(const std::vector<Tensor>& inputs, const Tensor &output,
        const Tensor &dEdy, int ii, Tensor &dEdxi);

    std::string type() { return "SquaredDistance"; }
};


class TanhNode: public Node {
  public:
    TanhNode(): Node() {}

    TanhNode(std::initializer_list<int> a): Node(a) {}

    ~TanhNode() {}

    void forward(const std::vector<Tensor>& inputs, Tensor &output);

    void backward(const std::vector<Tensor>& inputs, const Tensor &output,
        const Tensor &dEdy, int ii, Tensor &dEdxi);

    std::string type() { return "tanh"; }
};

class SigmoidNode: public Node {
  public:
    SigmoidNode(): Node() {}

    SigmoidNode(std::initializer_list<int> a): Node(a) {}

    ~SigmoidNode() {}

    void forward(const std::vector<Tensor>& inputs, Tensor &output);

    void backward(const std::vector<Tensor>& inputs, const Tensor &output,
        const Tensor &dEdy, int ii, Tensor &dEdxi);

    std::string type() { return "sigmoid"; }
};


class Embed: public Node {
  public:
    Embed(): Node() {}

    ~Embed() {}

    void forward(const std::vector<Tensor>& inputs, Tensor &output) {}

    void backward(const std::vector<Tensor>& inputs, const Tensor &output,
        const Tensor &dEdy, int ii, Tensor &dEdxi);

    std::string type() { return "Embed"; }
};

} // namespace rnnpp

#endif // RNNPP_NODE_H_
