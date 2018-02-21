#ifndef RNNPP_H_
#define RNNPP_H_

#include "dim.h"
#include "expr.h"
#include "graph.h"

namespace rnnpp {


Expression input(Graph &g, std::vector<float> &value);

Expression input(Graph &g, const Dim &dim, std::vector<float> &value);

Expression parameter(Graph &g, std::initializer_list<int> dim);

Expression squared_distance(const Expression &a, const Expression &b);


} // namespace rnnpp
#endif // RNNPP_H_
