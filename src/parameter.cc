#include <random>

#include "parameter.h"
#include "tensor.h"

namespace rnnpp {

void Initializer::init(Tensor &t) {
  std::random_device rnd;
  std::mt19937 mt(rnd());
  std::normal_distribution<> norm(0.0, 1.);
  for (int i=0; i < t.dim.size(); ++i) {
    t.data[i] = norm(mt);
  }
}

LookupParameter::LookupParameter(const Dim &dim) {
  all_values.dim = dim;
  all_values.data = new float[dim.size()];
  Initializer initializer;
  initializer.init(all_values);

  int num_words = dim.shape[0];
  int dim_emb = dim.shape[1];
  values.resize(num_words);
  for (int i=0; i < num_words; ++i) {
    values[i] = Tensor();
    values[i].dim = Dim({1, dim_emb});
    values[i].data = all_values.data + i * dim_emb;
  }
}

}
