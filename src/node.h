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
    Node(std::vector<int> a): args(a) {}

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


class ParameterNodeBase: public Node {
  public:
    virtual void add_gradient(const Tensor &dEdy)=0;
    std::string type() { return "ParameterNodeBase"; }
};

class ParameterNode: public ParameterNodeBase {
  public:
    ParameterNode(): ParameterNodeBase() {}

    ParameterNode(Parameter p): ParameterNodeBase() {
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

class LookupNode: public ParameterNodeBase {
  public:
    LookupNode(): ParameterNodeBase() {}

    LookupNode(LookupParameter p, int index): ParameterNodeBase(), index(index) {
      param = p;
      dim = p.all_values.dim;
    }

    void forward(const std::vector<Tensor>& inputs, Tensor &output);

    void backward(const std::vector<Tensor>& inputs, const Tensor &output,
        const Tensor &dEdy, int ii, Tensor &dEdxi);

    void add_gradient(const Tensor &dEdy) {
      std::cout << "add gradient at " << index << std::endl;
      param.grads[index] += dEdy;
    }

  private:
    LookupParameter param;
    int index;
};


/**
 *  y = [a, b]
 *  dEda = dEdy * dEda = dEdy[0: len(a)]
 *  dEdb = dEdy * dEdb = dEdy[len(a): len(b)]
 */
class Concat: public Node {
  public:
    Concat(): Node() {}

    Concat(std::initializer_list<int> a, int axis): Node(a), axis_(axis) {}
    Concat(std::vector<int> a, int axis): Node(a), axis_(axis) {}

    ~Concat(){}

    void forward(const std::vector<Tensor>& inputs, Tensor &output);
    void forward2(const std::vector<Tensor>& inputs, std::vector<Tensor> &output){};

    void backward(const std::vector<Tensor>& inputs, const Tensor &output,
        const Tensor &dEdy, int ii, Tensor &dEdxi);

    std::string type() { return "Concat"; }

  private:
    int axis_;
};
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

class Divide: public Node {
  public:
    Divide(): Node() {}

    Divide(std::initializer_list<int> a): Node(a) {}

    ~Divide(){}

    void forward(const std::vector<Tensor>& inputs, Tensor &output);

    void backward(const std::vector<Tensor>& inputs, const Tensor &output,
        const Tensor &dEdy, int ii, Tensor &dEdxi);

    std::string type() { return "Divide"; }
};

class DivideConst: public Node {
  public:
    DivideConst(): Node(), value(0) {}

    DivideConst(std::initializer_list<int> a): Node(a), value(0) {}

    DivideConst(std::initializer_list<int> a, int b, bool rhs_is_const)
      : Node(a), value(b), rhs_is_const(rhs_is_const) {}

    ~DivideConst(){}

    void forward(const std::vector<Tensor>& inputs, Tensor &output);

    void backward(const std::vector<Tensor>& inputs, const Tensor &output,
        const Tensor &dEdy, int ii, Tensor &dEdxi);

    std::string type() { return "DivideConst"; }
  private:
    float value;
    bool rhs_is_const;
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
