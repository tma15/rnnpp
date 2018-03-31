#include <algorithm>
#include <iostream>

#include "error.h"
#include "tensor.h"

namespace rnnpp {

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


void _sum(std::vector<int> &dst_index, 
    int pos, int axis, const Tensor &src, Tensor &dst) {

  if (pos == dst.dim.shape.size()-1) {
    for (dst_index[pos]=0; dst_index[pos] < dst.dim.shape[pos]; dst_index[pos] += 1) {

//      std::cout << "dst index:";
      int offset1 = 0;
      for (int k=0; k < dst_index.size(); ++k) {
//        std::cout << dst_index[k] << ",";
        offset1 += dst_index[k] * dst.dim.stride[k];
      }
//      std::cout << std::endl;
//      std::cout << "offset1:" << offset1 << std::endl;

      int j = 0;
//      std::cout << "src index:";
      int offset2base = 0;
      for (int k=0; k < src.dim.stride.size(); ++k) {
        if (k==axis) {
//          std::cout << "j" << ",";
          continue;
        }
//        std::cout << dst_index[j] << "(" << src.dim.stride[k] << "),";
        offset2base += dst_index[j] * src.dim.stride[k];
        j += 1;
      }
//      std::cout << std::endl;
//      std::cout << "offset2base:" << offset2base << std::endl;

      int ss = src.dim.size();
      int ds = dst.dim.size();
      for (int b=0; b < src.batch_size(); ++b) {
        dst.data[offset1 + b * ds] = 0;
        for (int i=0; i < src.dim.shape[axis]; ++i) {
          int offset2 = offset2base + i * src.dim.stride[axis];
          dst.data[offset1 + b * ds] += src.data[offset2 + b * ss];
  //        std::cout << " += src[" << offset2 << "] = " << src.data[offset2] << std::endl;
        }
  //      std::cout << "sum(offset1=" << offset1 << ") = " << dst.data[offset1] << std::endl;
  //      std::cout << std::endl;
  //      }
      }
    }
  } else {
    for (dst_index[pos]=0; dst_index[pos] < dst.dim.shape[pos]; dst_index[pos] += 1) {
      _sum(dst_index, pos+1, axis, src, dst);
    }
  }
}


void sum(const Tensor &src, Tensor &dst, int axis) {
  std::vector<int> dst_index(dst.dim.shape.size(), 0);
  _sum(dst_index, 0, axis, src, dst);
}

float sum(const Tensor &src) {
  float ret;
  for (int i=0; i < src.dim.size(); ++i) {
    ret += src.data[i];
  }
  return ret;
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
    std::vector<int> &cur, int pos, const Tensor &t, int indent, bool flag, int b) {
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
//      os << t.data[offset] << ",";
      os << t.data[offset + b * t.dim.size()] << ",";
    }

    int offset = 0;
    for (int k=0; k < cur.size(); ++k) {
      offset += cur[k] * stride[k];
    }
//    os << t.data[offset] << "]";
    os << t.data[offset + b * t.dim.size()] << "]";
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
      nest(os, shape, stride, cur, pos+1, t, indent+1, true, b);
    }
    nest(os, shape, stride, cur, pos+1, t, indent+1, false, b);
    os << "],";
    if (flag) {
      os << std::endl;
    }
  }
}

std::ostream& operator<<(std::ostream &os, const Tensor &t) {
  std::vector<int> cur(t.dim.shape.size(), 0);
  int level = 0;
  int indent = 0;
  bool newline_flag = false;

  for (int b=0; b < t.batch_size(); ++b) {
    nest(os, t.dim.shape, t.dim.stride, cur, level, t, indent, newline_flag, b);
  }
  return os;
}

} // namespace rnnpp
