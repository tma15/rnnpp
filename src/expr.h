#ifndef RNNPP_EXPR_H_
#define RNNPP_EXPR_H_

#include <initializer_list>

#include "graph.h"

namespace rnnpp {


class Expression {
  public:
    Expression(){}

    Expression(Graph* g, int i) : g_(g), id_(i) {}

    ~Expression(){}

    const Tensor& forward();

    std::vector<Tensor> forward2();

    void backward();

    void backward2();

    Graph* graph() { return g_; };

    /** 
     * Returns expression id in a graph
     */
    const int id() const { return id_; }

    Graph* g_;

  private:
    int id_;
};

float as_scalar(const Tensor& e);

Expression operator+(const Expression &a, const Expression &b);
Expression operator*(const Expression &a, const Expression &b);
Expression operator/(const Expression &a, const Expression &b);
Expression operator/(const Expression &a, float b);
Expression operator/(float a, const Expression &b);

Expression concat(const std::initializer_list<Expression> &xs, int axis);

std::vector<Expression> split(const Expression &x, int n, int axis);


} // namespace rnnpp

#endif // RNNPP_EXPR_H_
