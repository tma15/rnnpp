#include <iostream>

#include "node.h"

namespace rnnpp {

void InputNode::forward(const std::vector<Tensor> &inputs, Tensor &output) {
  output.dim = dim;
  output.data = const_cast<float*>(&data_->front());
}

void ParameterNode::forward(const std::vector<Tensor> &inputs, Tensor &output) {
  output.dim = dim;
  output.data = const_cast<float*>(data_.data());
}

void Mult::forward(const std::vector<Tensor> &inputs, Tensor &output) {
  output.dim = Dim({inputs[0].dim.shape[0]}, inputs[1].dim.batch_size);

  int k = output.dim.size() * output.dim.batch_size;
  output.data = new float[k];

//  output.dim.print_shape();
//  for (int i=0; i < inputs.size(); ++i) {
//    std::cout << "input:" << i << std::endl;
//    for (int b=0; b < inputs[i].dim.batch_size; ++b) {
//      for (int j=0; j < inputs[i].dim.size(); ++j) {
//        std::cout << inputs[i].data[j + b * inputs[i].dim.size()] << std::endl;
//      }
//      std::cout << "==" << std::endl;
//    }
//  }

  int out_w = output.dim.shape[0];

  Tensor w = inputs[0];
  Tensor x = inputs[1];

//  std::cout << "batchsize:" << x.dim.batch_size << std::endl;

  for(int b=0; b < x.dim.batch_size; ++b) {
    for (int j=0; j < x.dim.shape[0]; ++j) {
      for (int i=0; i < out_w; ++i) {
        int offset = out_w * j + i;
        output.data[i + b * output.dim.size()] += w.data[offset] * x.data[j + b * x.dim.size()];
      }
    }
  }

  for (int b=0; b < x.dim.batch_size; ++b) {
    for (int i=0; i < out_w; ++i) {
      std::cout << "out[" << i << "]" << " " << output.data[i + b * output.dim.size()] << std::endl;
    }
    std::cout << "==" << std::endl;
  }

}


void SquaredDistance::forward(const std::vector<Tensor> &inputs, Tensor &output) {
  output.dim = Dim({inputs[0].dim.shape[0]}, inputs[1].dim.batch_size);

  int k = output.dim.size() * output.dim.batch_size;
  output.data = new float[k];


  const Tensor &y1 = inputs[0];
  const Tensor &y2 = inputs[1];
  for (int b=0; b < output.dim.batch_size; ++b) {
    for (int i=0; i < output.dim.size(); ++i) {
      output.data[i + b * output.dim.size()] = sqrt(y1.data[i + b * y1.dim.size()] - y2.data[i + b * y2.dim.size()]);
    }
  }

  for (int b=0; b < output.dim.batch_size; ++b) {
    for (int i=0; i < output.dim.size(); ++i) {
      std::cout << "out[" << i << "]" << " " << output.data[i + b * output.dim.size()] << std::endl;
// f(y, y') = (y - y')^2
// dE/dy = dE/df * df/dy = dE/df * 2 * (y - y') * -1
// dE/dy' = dE/df * df/dy' = dE/df * 2 * (y - y') * 1
void SquaredDistance::backward(const std::vector<Tensor> &inputs, Tensor &output, int ii) {
  output.dim = Dim({inputs[ii].dim.shape[0]}, 1);
  int k = output.dim.size() * output.dim.batch_size;
  output.data = new float[k];

//  std::cout <<"backward SquaredDistance" << std::endl;
//  std::cout <<"output size:" << output.dim.size() << std::endl;
//  std::cout <<"#inputs:" << inputs.size() << std::endl;
//  std::cout <<"inputs dim:" << inputs[ii].dim.size() << std::endl;
//  std::cout <<"input index:" << ii << std::endl;

  if (ii == 0) {
    for (int b=0; b < inputs[1].dim.batch_size; ++b) {
      for (int i=0; i < output.dim.size(); ++i) {
        output.data[i] += 2. * (inputs[0].data[i + b * inputs[0].dim.size()] - \
            inputs[1].data[i + b * inputs[1].dim.size()]);
      }
    }
  } else if (ii == 1) {
    for (int b=0; b < inputs[0].dim.batch_size; ++b) {
      for (int i=0; i < output.dim.size(); ++i) {
        output.data[i] += -2. * (inputs[0].data[i + b * inputs[0].dim.size()] - \
            inputs[1].data[i + b * inputs[i].dim.size()]);
      }
    }
  }

//  for (int b=0; b < output.dim.batch_size; ++b) {
//    for (int i=0; i < output.dim.size(); ++i) {
//      std::cout << "g[" << i << "]" << " " << output.data[i + b * output.dim.size()] << std::endl;
//    }
//    std::cout << "==" << std::endl;
//  }

}

} // namespace rnnpp
