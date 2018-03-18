#ifndef RNNPP_EXPR_H_
#define RNNPP_EXPR_H_

#include "graph.h"

namespace rnnpp {


class Expression {
  public:
    Expression(){}

    Expression(Graph* g, int i) : g_(g), id_(i) {}

    ~Expression(){}

    const Tensor& forward();

    void backward();

    Graph* graph() { return g_; };

    int id() const { return id_; }

    Graph* g_;

  private:
    int id_;
};

float as_scalar(const Tensor& e);

Expression operator+(const Expression &a, const Expression &b);
Expression operator*(const Expression &a, const Expression &b);


} // namespace rnnpp

#endif // RNNPP_EXPR_H_
