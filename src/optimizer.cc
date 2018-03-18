#include <initializer_list>
#include <random>

#include "dim.h"
#include "expr.h"
#include "optimizer.h"
#include "parameter.h"

namespace rnnpp {

Parameter Optimizer::add_parameter(const std::initializer_list<int> &d) {
  Parameter p(d);
  std::random_device rnd;     // 非決定的な乱数生成器でシード生成機を生成
  std::mt19937 mt(rnd()); //
  std::normal_distribution<> norm(0.0, 0.1);       // 平均50, 分散値10の正規分布
  for (int i=0; i < p.data_.dim.size(); ++i) {
    p.data_.data[i] = norm(mt);
  }

  for (int i=0; i < p.grad_.dim.size(); ++i) {
    p.grad_.data[i] = 0.;
  }

  parameters_.push_back(&p);
  return p;
}

void Optimizer::update() {
  for (int i=0; i < parameters_.size(); ++i) {
//    std::cout << "Grad " << i << " " << parameters_[i]->grad_.dim << std::endl;
//    std::cout << parameters_[i]->grad_ << std::endl;
    for (int j=0; j < parameters_[i]->data_.dim.size(); ++j) {
      parameters_[i]->data_.data[j] -= 0.1 * parameters_[i]->grad_.data[j];
    }
    for (int j=0; j < parameters_[i]->grad_.dim.size(); ++j) {
      parameters_[i]->grad_.data[j] = 0.;
    }
  }
}

} // namespace rnnpp
