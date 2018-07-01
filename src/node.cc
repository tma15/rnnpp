#include <math.h>
#include <iostream>

#include "dim.h"
#include "error.h"
#include "expr.h"
#include "node.h"

namespace rnnpp {

void InputNode::forward(const std::vector<Tensor> &inputs, Tensor &output) {
  output.dim = dim;
  output.data = const_cast<float*>(&data_->front());
}

void InputNode::forward2(const std::vector<Tensor> &inputs,
    std::vector<Tensor*> &output) {
  output[0]->dim = dim;
  output[0]->data = const_cast<float*>(&data_->front());
}

void ParameterNode::forward(const std::vector<Tensor> &inputs, Tensor &output) {
  output.dim = dim;
  output.data = param.value.data;
}

void ParameterNode::forward2(const std::vector<Tensor> &inputs,
    std::vector<Tensor*> &output) {
  output[0]->dim = dim;
  output[0]->data = param.value.data;
}

void LookupNode::forward(const std::vector<Tensor> &inputs, Tensor &output) {
  output.dim = Dim({1, param.values[index].dim.shape[1]}, 1);
  output.data = param.values[index].data;
}

void LookupNode::forward2(const std::vector<Tensor> &inputs,
    std::vector<Tensor*> &output) {
  output[0]->dim = Dim({1, param.values[index].dim.shape[1]}, 1);
  output[0]->data = param.values[index].data;
}

void LookupNode::backward(const std::vector<Tensor> &inputs, const Tensor &output,
    const Tensor &dEdy, int ii, Tensor &dEdxi) {
  dEdxi.dim = Dim({1, param.values[index].dim.shape[1]}, 1);
  int k = dEdxi.dim.size() * dEdxi.dim.batch_size;
  dEdxi.data = new float[k];
  dEdxi = dEdy;
}

void LookupNode::backward2(const std::vector<Tensor> &inputs,
    const std::vector<Tensor> &output,
    const std::vector<Tensor> &dEdy, int ii, Tensor &dEdxi) {
  dEdxi.dim = Dim({1, param.values[index].dim.shape[1]}, 1);
  int k = dEdxi.dim.size() * dEdxi.dim.batch_size;
  dEdxi.data = new float[k];
  dEdxi = dEdy[0];
}

//void Square::forward(const std::vector<Tensor> &inputs, Tensor &output) {
//  RNNPP_CHECK(inputs.size() == 1, "Number of inputs is invalid: " << inputs.size());

//  int max_b = inputs[0].dim.batch_size;
//  output.dim = inputs[0].dim;
//  int k = output.dim.size() * output.dim.batch_size;
//  output.data = new float[k];

//  for (int b=0; b < max_b; ++b) {
//    for (int i=0; i < output.dim.size(); ++i) {
//      float v = inputs[0].data[i + b * inputs[0].dim.size()];
//      output.data[i + b * output.dim.size()] = v * v;
//    }
//  }
//}

//void Square::backward(const std::vector<Tensor> &inputs, const Tensor &output,
//    const Tensor &dEdy, int ii, Tensor &dEdxi) {
//}

void Sum::forward(const std::vector<Tensor> &inputs, Tensor &output) {
  int max_b = inputs[0].dim.batch_size;
  for (int i=1; i < inputs.size(); ++i) {
    if (inputs[i].dim.batch_size > max_b) max_b = inputs[i].dim.batch_size;
  }

  if (axis_ == -1) {
    output.dim = Dim({1, 1}, max_b);
  } else {
    std::vector<int> shape;
    for (int k=0; k < inputs[0].dim.shape.size(); ++k) {
      if (k == axis_) continue;
      shape.push_back(inputs[0].dim.shape[k]);
    }
    output.dim = Dim(shape, max_b);
  }

  int k = output.dim.size() * output.dim.batch_size;
  output.data = new float[k];

  output = Scalar(0.);
  for (int i=0; i < inputs.size(); ++i) {
    sum(inputs[i], output, axis_);
  }
}

void Sum::forward2(const std::vector<Tensor> &inputs, std::vector<Tensor*> &output) {
  int max_b = inputs[0].dim.batch_size;
  for (int i=1; i < inputs.size(); ++i) {
    if (inputs[i].dim.batch_size > max_b) max_b = inputs[i].dim.batch_size;
  }

  if (axis_ == -1) {
    output[0]->dim = Dim({1, 1}, max_b);
  } else {
    std::vector<int> shape;
    for (int k=0; k < inputs[0].dim.shape.size(); ++k) {
      if (k == axis_) continue;
      shape.push_back(inputs[0].dim.shape[k]);
    }
    output[0]->dim = Dim(shape, max_b);
  }

  int k = output[0]->dim.size() * output[0]->dim.batch_size;
  output[0]->data = new float[k];

  *output[0] = Scalar(0.);
  for (int i=0; i < inputs.size(); ++i) {
    sum(inputs[i], *output[0], axis_);
  }
}

void Sum::backward(const std::vector<Tensor> &inputs, const Tensor &output,
    const Tensor &dEdy, int ii, Tensor &dEdxi) {
  dEdxi.dim = inputs[ii].dim;
  int k = dEdxi.dim.size() * dEdxi.dim.batch_size;
  dEdxi.data = new float[k];
  dEdxi = Scalar(as_scalar(dEdy));
}

void Sum::backward2(const std::vector<Tensor> &inputs, const std::vector<Tensor> &output,
    const std::vector<Tensor> &dEdy, int ii, Tensor &dEdxi) {
  dEdxi.dim = inputs[ii].dim;
  int k = dEdxi.dim.size() * dEdxi.dim.batch_size;
  dEdxi.data = new float[k];
  dEdxi = Scalar(as_scalar(dEdy[0]));
}

void Concat::forward(const std::vector<Tensor> &inputs, Tensor &output) {
  if (axis_ == inputs[0].dim.shape.size()) { // concat along batch
    int b = 0;
    for (int i=0; i < inputs.size(); ++i) {
      b += inputs[i].dim.batch_size;
    }
    output.dim = Dim(inputs[0].dim.shape, b);
  } else { // concat along axis
    int b = inputs[0].dim.batch_size;

    std::vector<int> shape(inputs[0].dim.shape.size(), 0);
    int k = 0;
    for (int i=0; i < inputs[0].dim.shape.size(); ++i) {
      shape[k++] = inputs[0].dim.shape[i];
    }
    for (int i=1; i < inputs.size(); ++i) {
      shape[axis_] += inputs[i].dim.shape[axis_];
    }
    output.dim = Dim(shape, b);
  }

  int s = output.dim.size() * output.dim.batch_size;
  output.data = new float[s];
  concatenate(inputs, output, axis_);
}

void Concat::forward2(const std::vector<Tensor> &inputs, std::vector<Tensor*> &output) {
  if (axis_ == inputs[0].dim.shape.size()) { // concat along batch
    int b = 0;
    for (int i=0; i < inputs.size(); ++i) {
      b += inputs[i].dim.batch_size;
    }
    output[0]->dim = Dim(inputs[0].dim.shape, b);
  } else { // concat along axis
    int b = inputs[0].dim.batch_size;

    std::vector<int> shape(inputs[0].dim.shape.size(), 0);
    int k = 0;
    for (int i=0; i < inputs[0].dim.shape.size(); ++i) {
      shape[k++] = inputs[0].dim.shape[i];
    }
    for (int i=1; i < inputs.size(); ++i) {
      shape[axis_] += inputs[i].dim.shape[axis_];
    }
    output[0]->dim = Dim(shape, b);
  }

  int s = output[0]->dim.size() * output[0]->dim.batch_size;
  output[0]->data = new float[s];
  concatenate(inputs, *output[0], axis_);
}

void Concat::backward(const std::vector<Tensor> &inputs, const Tensor &output,
    const Tensor &dEdy, int ii, Tensor &dEdxi) {
  dEdxi.dim = inputs[ii].dim;
  int k = dEdxi.dim.size() * dEdxi.dim.batch_size;
  dEdxi.data = new float[k];
  slice(dEdy, dEdxi, ii, axis_);
}

void Concat::backward2(const std::vector<Tensor> &inputs, const std::vector<Tensor> &output,
    const std::vector<Tensor> &dEdy, int ii, Tensor &dEdxi) {
  dEdxi.dim = inputs[ii].dim;
  int k = dEdxi.dim.size() * dEdxi.dim.batch_size;
  dEdxi.data = new float[k];
  slice(dEdy[0], dEdxi, ii, axis_);
}

void Split::forward(const std::vector<Tensor> &inputs, Tensor &output) {
  if (axis_ == inputs[0].dim.shape.size()) { // concat along batch
  } else { // concat along axis
  }

  int s = output.dim.size() * output.dim.batch_size;
  output.data = new float[s];
  concatenate(inputs, output, axis_);
}

void Split::backward(const std::vector<Tensor> &inputs, const Tensor &output,
    const Tensor &dEdy, int ii, Tensor &dEdxi) {
  dEdxi.dim = inputs[ii].dim;
  int k = dEdxi.dim.size() * dEdxi.dim.batch_size;
  dEdxi.data = new float[k];
  slice(dEdy, dEdxi, ii, axis_);
}


// f(a, b) = a + b
// 
// dE/da = dEdy * dyda = dEdy
// dE/db = dEdy * dydb = dEdy
void Add::forward(const std::vector<Tensor> &inputs, Tensor &output) {
  RNNPP_CHECK(inputs.size() == 2, "Number of inputs is invalid: " << inputs.size());
  RNNPP_CHECK(inputs[0].dim == inputs[1].dim,
      "Invalid dimensions" << inputs[0].dim << " " << inputs[1].dim);
  int max_b = std::max(inputs[0].dim.batch_size, inputs[1].dim.batch_size);
  output.dim = Dim({inputs[0].dim.shape[0], inputs[1].dim.shape[1]}, max_b);
  int k = output.dim.size() * output.dim.batch_size;
  output.data = new float[k];

  Tensor a = inputs[0];
  Tensor b = inputs[1];
  output = a + b;
}

void Add::forward2(const std::vector<Tensor> &inputs,
    std::vector<Tensor*> &output) {
  RNNPP_CHECK(inputs.size() == 2, "Number of inputs is invalid: " << inputs.size());
  RNNPP_CHECK(inputs[0].dim == inputs[1].dim,
      "Invalid dimensions" << inputs[0].dim << " " << inputs[1].dim);
  int max_b = std::max(inputs[0].dim.batch_size, inputs[1].dim.batch_size);
  output[0]->dim = Dim({inputs[0].dim.shape[0], inputs[1].dim.shape[1]}, max_b);
  int k = output[0]->dim.size() * output[0]->dim.batch_size;
  output[0]->data = new float[k];

  Tensor a = inputs[0];
  Tensor b = inputs[1];
  *output[0] = a + b;
}

void Add::backward(const std::vector<Tensor> &inputs, const Tensor &output,
    const Tensor &dEdy, int ii, Tensor &dEdxi) {
  dEdxi.dim = inputs[ii].dim;
  int k = dEdxi.dim.size() * dEdxi.dim.batch_size;
  dEdxi.data = new float[k];
  dEdxi = dEdy;
}

void Add::backward2(const std::vector<Tensor> &inputs, const std::vector<Tensor> &output,
    const std::vector<Tensor> &dEdy, int ii, Tensor &dEdxi) {
  dEdxi.dim = inputs[ii].dim;
  int k = dEdxi.dim.size() * dEdxi.dim.batch_size;
  dEdxi.data = new float[k];
  dEdxi = dEdy[0];
}

void Mult::forward(const std::vector<Tensor> &inputs, Tensor &output) {
  RNNPP_CHECK(inputs.size() == 2, "Number of inputs is invalid: " << inputs.size());

  int max_b = std::max(inputs[0].dim.batch_size, inputs[1].dim.batch_size);
  output.dim = Dim({inputs[0].dim.shape[0], inputs[1].dim.shape[1]}, max_b);
  int k = output.dim.size() * output.dim.batch_size;
  output.data = new float[k];

  Tensor w = inputs[0];
  Tensor x = inputs[1];
  matmul(w, x, output);
}

void Mult::forward2(const std::vector<Tensor> &inputs,
    std::vector<Tensor*> &output) {
  RNNPP_CHECK(inputs.size() == 2, "Number of inputs is invalid: " << inputs.size());

  int max_b = std::max(inputs[0].dim.batch_size, inputs[1].dim.batch_size);
  output[0]->dim = Dim({inputs[0].dim.shape[0], inputs[1].dim.shape[1]}, max_b);
  int k = output[0]->dim.size() * output[0]->dim.batch_size;
  output[0]->data = new float[k];

  Tensor w = inputs[0];
  Tensor x = inputs[1];
  matmul(w, x, *output[0]);
}

// f(w, x) = w * x  (N, B) = (N, M) x (M, B)
//
// dE/dw = dE/df * df/dw = dE/df * x    (N, M) = (N, 1) x (1, M)
// dE/dx = dE/df * df/dw = dE/df * w    (M, 1) = {(N, 1)^T x (N, M)}^T
void Mult::backward(const std::vector<Tensor> &inputs, const Tensor &output,
    const Tensor &dEdy, int ii, Tensor &dEdxi) {
  dEdxi.dim = inputs[ii].dim;
  int k = dEdxi.dim.size() * dEdxi.dim.batch_size;
  dEdxi.data = new float[k];

  Tensor w = inputs[0];
  Tensor x = inputs[1];

//  std::cout << "Mult::backward:" << std::endl;
//  std::cout << "w" << w.dim << ":\n" << w << std::endl;
//  std::cout << "x" << x.dim << ":\n" << x << std::endl;
//  std::cout << "dEdxi" << dEdxi.dim << ":\n" << dEdxi << std::endl;
  if (ii == 0) {
    matmul(dEdy, x.transpose(), dEdxi);
//    std::cout << dEdxi.dim << " = " << dEdy.dim << " x " << x.transpose().dim << std::endl;
//    std::cout << "dEdy" << std::endl;
//    std::cout << dEdy << std::endl;
//    std::cout << "x.T" << std::endl;
//    std::cout << x.transpose() << std::endl;
//    std::cout << "dEdw" << std::endl;
//    std::cout << dEdxi << std::endl;
  } else {
    matmul(w.transpose(), dEdy, dEdxi);
//    std::cout << "dEdy" << std::endl;
//    std::cout << dEdy << std::endl;
//    std::cout << "w" << std::endl;
//    std::cout << w.transpose() << std::endl;
//    std::cout << "dEdx" << std::endl;
//    std::cout << dEdxi << std::endl;
  }
//  std::cout << type() << std::endl;
//  std::cout << "dEdx: " << dEdxi.dim << ":\n" << dEdxi << std::endl;
}

void Mult::backward2(const std::vector<Tensor> &inputs, const std::vector<Tensor> &output,
    const std::vector<Tensor> &dEdy, int ii, Tensor &dEdxi) {
  dEdxi.dim = inputs[ii].dim;
  int k = dEdxi.dim.size() * dEdxi.dim.batch_size;
  dEdxi.data = new float[k];

  Tensor w = inputs[0];
  Tensor x = inputs[1];

//  std::cout << "Mult::backward:" << std::endl;
//  std::cout << "w" << w.dim << ":\n" << w << std::endl;
//  std::cout << "x" << x.dim << ":\n" << x << std::endl;
//  std::cout << "dEdxi" << dEdxi.dim << ":\n" << dEdxi << std::endl;
  if (ii == 0) {
    matmul(dEdy[0], x.transpose(), dEdxi);
//    std::cout << dEdxi.dim << " = " << dEdy.dim << " x " << x.transpose().dim << std::endl;
//    std::cout << "dEdy" << std::endl;
//    std::cout << dEdy << std::endl;
//    std::cout << "x.T" << std::endl;
//    std::cout << x.transpose() << std::endl;
//    std::cout << "dEdw" << std::endl;
//    std::cout << dEdxi << std::endl;
  } else {
    matmul(w.transpose(), dEdy[0], dEdxi);
//    std::cout << "dEdy" << std::endl;
//    std::cout << dEdy << std::endl;
//    std::cout << "w" << std::endl;
//    std::cout << w.transpose() << std::endl;
//    std::cout << "dEdx" << std::endl;
//    std::cout << dEdxi << std::endl;
  }
//  std::cout << type() << std::endl;
//  std::cout << "dEdx: " << dEdxi.dim << ":\n" << dEdxi << std::endl;
}


void Divide::forward(const std::vector<Tensor> &inputs, Tensor &output) {
  RNNPP_CHECK(inputs.size() == 2, "Number of inputs is invalid: " << inputs.size());
  RNNPP_CHECK(inputs[0].dim == inputs[1].dim,
      "Invalid dimensions" << inputs[0].dim << " " << inputs[1].dim);

  int max_b = std::max(inputs[0].dim.batch_size, inputs[1].dim.batch_size);
  output.dim = Dim(inputs[0].dim.shape, max_b);
  int k = output.dim.size() * output.dim.batch_size;
  output.data = new float[k];

  Tensor a = inputs[0];
  Tensor b = inputs[1];
  output = a / b;
}

void Divide::forward2(const std::vector<Tensor> &inputs, std::vector<Tensor*> &output) {
  RNNPP_CHECK(inputs.size() == 2, "Number of inputs is invalid: " << inputs.size());
  RNNPP_CHECK(inputs[0].dim == inputs[1].dim,
      "Invalid dimensions" << inputs[0].dim << " " << inputs[1].dim);

  int max_b = std::max(inputs[0].dim.batch_size, inputs[1].dim.batch_size);
  output[0]->dim = Dim(inputs[0].dim.shape, max_b);
  int k = output[0]->dim.size() * output[0]->dim.batch_size;
  output[0]->data = new float[k];

  Tensor a = inputs[0];
  Tensor b = inputs[1];
  *output[0] = a / b;
}

// y = a / b
// dEda = dEdy * dyda = dEdy * (1/b)
// dEdb = dEdy * dydb = dEdy * (-a / b^2)
void Divide::backward(const std::vector<Tensor> &inputs, const Tensor &output,
    const Tensor &dEdy, int ii, Tensor &dEdxi) {
  dEdxi.dim = inputs[ii].dim;
  int k = dEdxi.dim.size() * dEdxi.dim.batch_size;
  dEdxi.data = new float[k];

  if (ii == 0) {
    dEdxi = dEdy / inputs[1];
  } else {
    dEdxi = dEdy * (-inputs[0] / square(inputs[1]));
  }
}

void Divide::backward2(const std::vector<Tensor> &inputs, const std::vector<Tensor> &output,
    const std::vector<Tensor> &dEdy, int ii, Tensor &dEdxi) {
  dEdxi.dim = inputs[ii].dim;
  int k = dEdxi.dim.size() * dEdxi.dim.batch_size;
  dEdxi.data = new float[k];

  if (ii == 0) {
    dEdxi = dEdy[0] / inputs[1];
  } else {
    dEdxi = dEdy[0] * (-inputs[0] / square(inputs[1]));
  }
}

void DivideConst::forward(const std::vector<Tensor> &inputs, Tensor &output) {
  RNNPP_CHECK(inputs.size() == 1, "Number of inputs is invalid: " << inputs.size());

  int max_b = inputs[0].dim.batch_size;
  output.dim = Dim(inputs[0].dim.shape, max_b);
  int k = output.dim.size() * output.dim.batch_size;
  output.data = new float[k];

  Tensor a = inputs[0];
  if (rhs_is_const) {
    output = a / Scalar(value);
  } else {
    output = Scalar(value) / a;
  }
}

void DivideConst::forward2(const std::vector<Tensor> &inputs, std::vector<Tensor*> &output) {
  RNNPP_CHECK(inputs.size() == 1, "Number of inputs is invalid: " << inputs.size());

  int max_b = inputs[0].dim.batch_size;
  output[0]->dim = Dim(inputs[0].dim.shape, max_b);
  int k = output[0]->dim.size() * output[0]->dim.batch_size;
  output[0]->data = new float[k];

  Tensor a = inputs[0];
  if (rhs_is_const) {
    *output[0] = a / Scalar(value);
  } else {
    *output[0] = Scalar(value) / a;
  }
}

void DivideConst::backward(const std::vector<Tensor> &inputs, const Tensor &output,
    const Tensor &dEdy, int ii, Tensor &dEdxi) {
  dEdxi.dim = inputs[ii].dim;
  int k = dEdxi.dim.size() * dEdxi.dim.batch_size;
  dEdxi.data = new float[k];

  if (rhs_is_const) {
    dEdxi = dEdy / Scalar(value);
  } else {
    dEdxi = dEdy * (-Scalar(value) / square(inputs[0]));
  }
}

void DivideConst::backward2(const std::vector<Tensor> &inputs, const std::vector<Tensor> &output,
    const std::vector<Tensor> &dEdy, int ii, Tensor &dEdxi) {
  dEdxi.dim = inputs[ii].dim;
  int k = dEdxi.dim.size() * dEdxi.dim.batch_size;
  dEdxi.data = new float[k];

  if (rhs_is_const) {
    dEdxi = dEdy[0] / Scalar(value);
  } else {
    dEdxi = dEdy[0] * (-Scalar(value) / square(inputs[0]));
  }
}


void TanhNode::forward(const std::vector<Tensor> &inputs, Tensor &output) {
  output.dim = inputs[0].dim;
  int k = output.dim.size() * output.dim.batch_size;
  output.data = new float[k];
  output = (exp(inputs[0]) - exp(-inputs[0])) / (exp(inputs[0]) + exp(-inputs[0]));
}

void TanhNode::forward2(const std::vector<Tensor> &inputs,
    std::vector<Tensor*> &output) {
  RNNPP_CHECK(output.size() == 1, "Number of output must be 1");
  output[0]->dim = inputs[0].dim;
  int k = output[0]->dim.size() * output[0]->dim.batch_size;
  output[0]->data = new float[k];
  *output[0] = (exp(inputs[0]) - exp(-inputs[0])) / (exp(inputs[0]) + exp(-inputs[0]));
}

void TanhNode::backward(const std::vector<Tensor> &inputs, const Tensor &output,
    const Tensor &dEdy, int ii, Tensor &dEdxi) {
  dEdxi.dim = inputs[ii].dim;
  int k = dEdxi.dim.size() * dEdxi.dim.batch_size;
  dEdxi.data = new float[k];
  dEdxi = dEdy * (Scalar(1.) - (output * output));
}

void TanhNode::backward2(const std::vector<Tensor> &inputs, const std::vector<Tensor> &output,
    const std::vector<Tensor> &dEdy, int ii, Tensor &dEdxi) {
  dEdxi.dim = inputs[ii].dim;
  int k = dEdxi.dim.size() * dEdxi.dim.batch_size;
  dEdxi.data = new float[k];
  dEdxi = dEdy[0] * (Scalar(1.) - (output[0] * output[0]));
}

void SigmoidNode::forward(const std::vector<Tensor> &inputs, Tensor &output) {
  output.dim = inputs[0].dim;
  int k = output.dim.size() * output.dim.batch_size;
  output.data = new float[k];
  output = Scalar(1.) / (Scalar(1.) + exp(-inputs[0]));
}

void SigmoidNode::forward2(const std::vector<Tensor> &inputs, std::vector<Tensor*> &output) {
  output[0]->dim = inputs[0].dim;
  int k = output[0]->dim.size() * output[0]->dim.batch_size;
  output[0]->data = new float[k];
  *output[0] = Scalar(1.) / (Scalar(1.) + exp(-inputs[0]));
}

void SigmoidNode::backward(const std::vector<Tensor> &inputs, const Tensor &output,
    const Tensor &dEdy, int ii, Tensor &dEdxi) {
  dEdxi.dim = inputs[0].dim;
  int k = dEdxi.dim.size() * dEdxi.dim.batch_size;
  dEdxi.data = new float[k];
  dEdxi = dEdy * (Scalar(1.) - output) * output;
}

void SigmoidNode::backward2(const std::vector<Tensor> &inputs, const std::vector<Tensor> &output,
    const std::vector<Tensor> &dEdy, int ii, Tensor &dEdxi) {
  dEdxi.dim = inputs[0].dim;
  int k = dEdxi.dim.size() * dEdxi.dim.batch_size;
  dEdxi.data = new float[k];
  dEdxi = dEdy[0] * (Scalar(1.) - output[0]) * output[0];
}

// f(y, y') = (y - y')^2
void SquaredDistance::forward(const std::vector<Tensor> &inputs, Tensor &output) {
  RNNPP_CHECK(inputs.size() == 2, "Number of inputs is invalid: " << inputs.size());

  int max_b = std::max(inputs[0].dim.batch_size, inputs[1].dim.batch_size);

  output.dim = Dim(inputs[0].dim.shape, max_b);
  int k = output.dim.size() * output.dim.batch_size;
  output.data = new float[k];

  const Tensor &y1 = inputs[0];
  const Tensor &y2 = inputs[1];

  output = square(y1 - y2);
}

// f(y, y') = (y - y')^2
void SquaredDistance::forward2(const std::vector<Tensor> &inputs,
    std::vector<Tensor*> &output) {
  RNNPP_CHECK(inputs.size() == 2, "Number of inputs is invalid: " << inputs.size());

  int max_b = std::max(inputs[0].dim.batch_size, inputs[1].dim.batch_size);

  output[0]->dim = Dim(inputs[0].dim.shape, max_b);
  int k = output[0]->dim.size() * output[0]->dim.batch_size;
  output[0]->data = new float[k];

  const Tensor &y1 = inputs[0];
  const Tensor &y2 = inputs[1];

  *output[0] = square(y1 - y2);
}

// dE/dy = dE/df * df/dy = dE/df * 2 * (y - y') * 1
// dE/dy' = dE/df * df/dy' = dE/df * 2 * (y - y') * -1
void SquaredDistance::backward(const std::vector<Tensor> &inputs, const Tensor &output,
    const Tensor &dEdy, int ii, Tensor &dEdxi) {
  dEdxi.dim = inputs[ii].dim;
  int k = dEdxi.dim.size() * dEdxi.dim.batch_size;
  dEdxi.data = new float[k];

  if (ii == 0) {
    dEdxi = dEdy * Scalar(2.) * (inputs[0] - inputs[1]);
  } else if (ii == 1) {
    dEdxi = dEdy * Scalar(-2.) * (inputs[0] - inputs[1]);
  }
//  std::cout << "SquaredDistance dEdx:" << output.dim << std::endl;
//  std::cout << dEdxi << std::endl;
}

void SquaredDistance::backward2(const std::vector<Tensor> &inputs,
    const std::vector<Tensor> &output,
    const std::vector<Tensor> &dEdy, int ii, Tensor &dEdxi) {
  dEdxi.dim = inputs[ii].dim;
  int k = dEdxi.dim.size() * dEdxi.dim.batch_size;
  dEdxi.data = new float[k];

  if (ii == 0) {
    dEdxi = dEdy[0] * Scalar(2.) * (inputs[0] - inputs[1]);
  } else if (ii == 1) {
    dEdxi = dEdy[0] * Scalar(-2.) * (inputs[0] - inputs[1]);
  }
//  std::cout << "SquaredDistance dEdx:" << output.dim << std::endl;
//  std::cout << dEdxi << std::endl;
}


} // namespace rnnpp
