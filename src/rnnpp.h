#ifndef RNNPP_H_
#define RNNPP_H_

#include "dim.h"
#include "expr.h"
#include "graph.h"

namespace rnnpp {

Expression input(Graph &g, std::vector<float> &value);

Expression input(Graph &g, const Dim &dim, std::vector<float> &value);

Expression parameter(Graph &g, const Parameter &p);

Expression lookup(Graph &g, const LookupParameter &lp, int index);

Expression squared_distance(const Expression &a, const Expression &b);

Expression sum(const Expression &x);

Expression tanh(const Expression &x);

Expression sigmoid(const Expression &x);


} // namespace rnnpp
#endif // RNNPP_H_
