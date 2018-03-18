#ifndef RNNPP_OPTIMIZER_H_
#define RNNPP_OPTIMIZER_H_

#include <initializer_list>

#include "expr.h"
#include "parameter.h"

namespace rnnpp {

class Optimizer {
  public:
    Optimizer() {}
    ~Optimizer() {}

    Parameter add_parameter(const std::initializer_list<int> &d);

    void update();

  protected:
    std::vector<Parameter*> parameters_;
};

} // namespace rnnpp

#endif // RNNPP_OPTIMIZER_H_
