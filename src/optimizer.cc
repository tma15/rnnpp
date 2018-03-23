#include <initializer_list>
#include <random>

#include "dim.h"
#include "expr.h"
#include "optimizer.h"
#include "parameter.h"

namespace rnnpp {

Parameter Optimizer::add_parameter(const std::initializer_list<int> &d) {
  Parameter p(d);
  Initializer initializer;
  initializer.init(p.value);
  p.grad = Scalar(0.);

  parameters_.push_back(&p);
  return p;
}

void Optimizer::update() {
  for (int i=0; i < parameters_.size(); ++i) {
    parameters_[i]->value -= Scalar(0.1) * parameters_[i]->grad;
    parameters_[i]->grad = Scalar(0.);
  }
}

} // namespace rnnpp
