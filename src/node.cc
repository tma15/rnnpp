#include <math.h>
#include <iostream>

#include "error.h"
#include "node.h"

namespace rnnpp {

void InputNode::forward(const std::vector<Tensor> &inputs, Tensor &output) {
  output.dim = dim;
  output.data = const_cast<float*>(&data_->front());
}

void ParameterNode::forward(const std::vector<Tensor> &inputs, Tensor &output) {
  output.dim = dim;
  output.data = param.value.data;
//  std::cout << "ParameterNode::forward dim:" << dim << std::endl;
//  std::cout << output << std::endl;
//  std::cout << param.data_.dim.shape.size() << std::endl;
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
  output.dim = inputs[0].dim;
  int k = output.dim.size() * output.dim.batch_size;
  output.data = new float[k];

  for (int i=0; i < inputs.size(); ++i) {
    output += inputs[i];
  }
}

void Sum::backward(const std::vector<Tensor> &inputs, const Tensor &output,
    const Tensor &dEdy, int ii, Tensor &dEdxi) {
  dEdxi.dim = Dim(inputs[ii].dim.shape, 1);
  int k = dEdxi.dim.size() * dEdxi.dim.batch_size;
  dEdxi.data = new float[k];
  dEdxi = dEdy;
}


// f(a, b) = a + b
// 
// dE/da = dEdy * dyda = dEdy
// dE/db = dEdy * dydb = dEdy
void Add::forward(const std::vector<Tensor> &inputs, Tensor &output) {
  RNNPP_CHECK(inputs.size() == 2, "Number of inputs is invalid: " << inputs.size());

  int max_b = std::max(inputs[0].dim.batch_size, inputs[1].dim.batch_size);
  output.dim = Dim({inputs[0].dim.shape[0], inputs[1].dim.shape[1]}, max_b);
  int k = output.dim.size() * output.dim.batch_size;
  output.data = new float[k];

  Tensor a = inputs[0];
  Tensor b = inputs[1];
  output = a + b;
//  std::cout << "add forward" << std::endl;
//  std::cout << "a:" << a << std::endl;
//  std::cout << "b:" << b << std::endl;
//  std::cout << "y:" << output << std::endl;
}

void Add::backward(const std::vector<Tensor> &inputs, const Tensor &output,
    const Tensor &dEdy, int ii, Tensor &dEdxi) {

  dEdxi.dim = Dim(inputs[ii].dim.shape, 1);
  int k = dEdxi.dim.size() * dEdxi.dim.batch_size;
  dEdxi.data = new float[k];

  dEdxi = dEdy;
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

//  std::cout << "matmul forward" << std::endl;
//  std::cout << "w" << w.dim << ":\n" << w << std::endl;
//  std::cout << "x" << x.dim << ":\n" << x << std::endl;
//  std::cout << "y" << output.dim << ":\n" << output << std::endl;
}

// f(w, x) = w * x  (N, B) = (N, M) x (M, B)
//
// dE/dw = dE/df * df/dw = dE/df * x    (N, M) = (N, 1) x (1, M)
// dE/dx = dE/df * df/dw = dE/df * w    (M, 1) = {(N, 1)^T x (N, M)}^T
void Mult::backward(const std::vector<Tensor> &inputs, const Tensor &output,
    const Tensor &dEdy, int ii, Tensor &dEdxi) {
  dEdxi.dim = Dim(inputs[ii].dim.shape, 1);
  int k = dEdxi.dim.size() * dEdxi.dim.batch_size;
  dEdxi.data = new float[k];

  Tensor w = inputs[0];
  Tensor x = inputs[1];

//  std::cout << "Mult::backward:" << std::endl;
//  std::cout << "w" << w.dim << ":\n" << w << std::endl;
//  std::cout << "x" << x.dim << ":\n" << x << std::endl;
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

void TanhNode::forward(const std::vector<Tensor> &inputs, Tensor &output) {
  output.dim = inputs[0].dim;
  int k = output.dim.size() * output.dim.batch_size;
  output.data = new float[k];

  output = (exp(inputs[0]) - exp(-inputs[0])) / (exp(inputs[0]) + exp(-inputs[0]));

//  std::cout << "Tanh:" << std::endl;
//  std::cout << "x:" << std::endl;
//  std::cout << inputs[0] << std::endl;
//  std::cout << "y:" << std::endl;
//  std::cout << output << std::endl;
}

void TanhNode::backward(const std::vector<Tensor> &inputs, const Tensor &output,
    const Tensor &dEdy, int ii, Tensor &dEdxi) {
  dEdxi.dim = Dim(inputs[0].dim.shape, 1);
  int k = dEdxi.dim.size() * dEdxi.dim.batch_size;
  dEdxi.data = new float[k];

  dEdxi = dEdy * (Scalar(1.) - (output * output));
//  std::cout << type() << std::endl;
//  std::cout << "dEdx: " << dEdxi.dim << ":\n" << dEdxi << std::endl;
}

void SigmoidNode::forward(const std::vector<Tensor> &inputs, Tensor &output) {
  output.dim = inputs[0].dim;
  int k = output.dim.size() * output.dim.batch_size;
  output.data = new float[k];

  output = Scalar(1.) / (Scalar(1.) + exp(-inputs[0]));

//  std::cout << type() << std::endl;
//  std::cout << "x:" << std::endl;
//  std::cout << inputs[0] << std::endl;
//  std::cout << "y:" << std::endl;
//  std::cout << output << std::endl;
}

void SigmoidNode::backward(const std::vector<Tensor> &inputs, const Tensor &output,
    const Tensor &dEdy, int ii, Tensor &dEdxi) {
  dEdxi.dim = Dim(inputs[0].dim.shape, 1);
  int k = dEdxi.dim.size() * dEdxi.dim.batch_size;
  dEdxi.data = new float[k];

  dEdxi = dEdy * (Scalar(1.) - output) * output;
//  std::cout << type() << std::endl;
//  std::cout << "dEdx: " << dEdxi.dim << ":\n" << dEdxi << std::endl;
}



// f(y, y') = (y - y')^2
void SquaredDistance::forward(const std::vector<Tensor> &inputs, Tensor &output) {
  RNNPP_CHECK(inputs.size() == 2, "Number of inputs is invalid: " << inputs.size());

  int max_b = std::max(inputs[0].dim.batch_size, inputs[1].dim.batch_size);

  output.dim = Dim({inputs[0].dim[0], inputs[0].dim[1]}, max_b);
  int k = output.dim.size() * output.dim.batch_size;
  output.data = new float[k];

  const Tensor &y1 = inputs[0];
  const Tensor &y2 = inputs[1];

  output = square(y1 - y2);
//  std::cout << "y1:" << y1 << " y2:" << y2 << std::endl;
//  std::cout << "SquaredDistance:" << output.dim << std::endl;
//  std::cout << output << std::endl;
}

// dE/dy = dE/df * df/dy = dE/df * 2 * (y - y') * 1
// dE/dy' = dE/df * df/dy' = dE/df * 2 * (y - y') * -1
void SquaredDistance::backward(const std::vector<Tensor> &inputs, const Tensor &output,
    const Tensor &dEdy, int ii, Tensor &dEdxi) {
  dEdxi.dim = Dim({inputs[ii].dim.shape[0], 1}, 1);
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

} // namespace rnnpp
