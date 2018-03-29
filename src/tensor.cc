#include <algorithm>
#include <iostream>

#include "error.h"
#include "tensor.h"

namespace rnnpp {

//std::ostream& operator<<(std::ostream &os, const Tensor &t) {
//  RNNPP_CHECK(t.dim.shape.size() == 2, "Tensor must be two dimension");

  //   offset = i * stride[0] + j * stride[1] + k * stride[2] + ...

//  int size = t.dim.size();
//  for (int b=0; b < t.dim.batch_size; ++b) {
//    os << "[";
//    for (int i=0; i < t.dim[0]; ++i) {
//      if (i > 0) {
//        std::cout << " ";
//      }
//      os << "[";
//      for (int j=0; j < t.dim[1]-1; ++j) {
//        os << t(i, j) << ",";
//        int offset = i * t.dim.stride[0] + j * t.dim.stride[1] + b * t.dim.size();;
//        os << t.data[offset] << ", ";
//      }
//      os << t(i, t.dim.shape[1]-1) << "]";
//      int offset = i * t.dim.stride[0] + (t.dim.shape[1]-1) * t.dim.stride[1] + b * t.dim.size();
//      os << t.data[offset] << "]";
//      if (i < t.dim[0]-1) {
//        os << "," << std::endl;
//      }
//    }
//    os << "]";
//    if (b != t.dim.batch_size -1) {
//      os << "," << std::endl;
//    }
//  }
//  return os;
//}

Tensor Tensor::transpose() {
  Tensor dest;

  std::vector<int> d(dim.shape.size());
  std::vector<int> s(dim.stride.size());

  // Example: [2, 1, 5, 3, 4] ==>  [4, 3, 5, 1, 2]
  for (int i=0; i < dim.shape.size() / 2 + dim.shape.size() % 2; ++i) {
    d[i] = dim[dim.shape.size() - 1 - i];
    d[dim.shape.size() - 1 - i] = dim[i];

    s[i] = dim.stride[dim.stride.size() - 1 - i];
    s[dim.stride.size() - 1 - i] = dim.stride[i];
  }
  dest.dim = Dim(d, dim.batch_size);
  dest.dim.stride = s;

  dest.data = data;
  return dest;
}

// (M, N) = (M, K) x (K, N)
void matmul(const Tensor &lhs, const Tensor &rhs, Tensor &dest) {
  int M = dest.dim[0];
  int N = dest.dim[1];
  int K = lhs.dim[1];
  int max_b = std::max(lhs.dim.batch_size, rhs.dim.batch_size);

  RNNPP_CHECK(lhs.dim[1] == rhs.dim[0], "Invalid dimension in matmul");
  RNNPP_CHECK(dest.dim[0] == lhs.dim[0], "Invalid dimension in matmul");
  RNNPP_CHECK(dest.dim[1] == rhs.dim[1], "Invalid dimension in matmul");

  const float* ld = lhs.cdata();
  const float* rd = rhs.cdata();
  float* destd = dest.data;

  for (int i=0; i < dest.dim.size(); ++i) {
    destd[i] = 0.;
  }

  for (int b=0; b < max_b; ++b) {
    for (int m=0; m < M; ++m) {
      for (int n=0; n < N; ++n) {
        int offset1 = m * dest.dim.stride[0] + n * dest.dim.stride[1];
        for (int k=0; k < K; ++k) {
          int offset2 = m * lhs.dim.stride[0] + k * lhs.dim.stride[1];
          int offset3 = k * rhs.dim.stride[0] + n * rhs.dim.stride[1];

//          float l = ld[lhs.dim.size() * b % lhs.dim.batch_size + offset2 + k];
//          float r = rd[rhs.dim.size() * b % rhs.dim.batch_size + offset3 + n];
//          std::cout << "["<<m<<","<<n<<"]"<< " += [" <<m<<","<<k<<"]" << l << " * "\
//                    << "["<<k<<","<<n<<"]"<< r << std::endl;

//          destd[offset1 + n] += ld[lhs.dim.size() * b % lhs.dim.batch_size + offset2 + k] \
//                                * rd[rhs.dim.size() * b % rhs.dim.batch_size + offset3 + n];
          destd[offset1] += ld[lhs.dim.size() * b % lhs.dim.batch_size + offset2] \
                                * rd[rhs.dim.size() * b % rhs.dim.batch_size + offset3];
        }
      }
    }
  }
}

void nest(std::ostream &os, const std::vector<int> &shape, const std::vector<int> &stride,
    std::vector<int> &cur, int pos, const Tensor &t, int indent, bool flag) {
  if (pos == shape.size()-1) {

    os << "[";
    for (cur[pos]=0; cur[pos] < shape[pos]-1; cur[pos] += 1) {
//      std::cout << "index:";

      int offset = 0;
      for (int k=0; k < cur.size(); ++k) {
//        std::cout << cur[k] << ",";
        offset += cur[k] * stride[k];
      }
//      std::cout << cur.back() << ":";
      os << t.data[offset] << ",";
    }

    int offset = 0;
    for (int k=0; k < cur.size(); ++k) {
      offset += cur[k] * stride[k];
    }
    os << t.data[offset] << "]";
    if (flag) {
//      std::cout << "newline:" ;
      os << "," << std::endl;
//      for (int i=0; i < indent; ++i) {
//        os << " ";
//      }
    }

  } else {
//    for (int i=0; i < indent; ++i) {
//      std::cout << " ";
//    }

    os << "[";
    for (cur[pos]=0; cur[pos] < shape[pos]-1; cur[pos] += 1) {
      nest(os, shape, stride, cur, pos+1, t, indent+1, true);
    }
    nest(os, shape, stride, cur, pos+1, t, indent+1, false);
    os << "],";
    if (flag) {
      os << std::endl;
    }
  }
}

std::ostream& operator<<(std::ostream &os, const Tensor &t) {
//void rec_loop(const Tensor &t) {
//    std::cout << "shape:";
//    for (int k=0; k < t.dim.shape.size()-1; ++k) {
//      std::cout << t.dim.shape[k] << ",";
//    }
//    std::cout << t.dim.shape.back() << std::endl;

//    std::cout << "stride:";
//    for (int k=0; k < t.dim.stride.size()-1; ++k) {
//      std::cout << t.dim.stride[k] << ",";
//    }
//    std::cout << t.dim.stride.back() << std::endl;

  std::vector<int> cur(t.dim.shape.size(), 0);
  int level = 0;
  int indent = 0;
  bool newline_flag = false;
  nest(os, t.dim.shape, t.dim.stride, cur, level, t, indent, newline_flag);
//  os << std::endl;
  return os;
}

} // namespace rnnpp
