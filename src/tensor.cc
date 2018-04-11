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


Tensor Tensor::batch_elem(int bid) {
  Tensor t;
  t.dim = Dim(dim.shape);
  t.data = data + bid * dim.size();
  return t;
}


void _sum(std::vector<int> &dst_index, 
    int pos, int axis, const Tensor &src, Tensor &dst) {

  int ss = src.dim.size();
  int ds = dst.dim.size();

  if (pos == dst.dim.shape.size()-1) {
    for (dst_index[pos]=0; dst_index[pos] < dst.dim.shape[pos]; dst_index[pos] += 1) {

      int offset1 = 0;
      if (axis > -1) {
        for (int k=0; k < dst_index.size(); ++k) {
          offset1 += dst_index[k] * dst.dim.stride[k];
//          std::cout << dst_index[k] << ",";
        }
//        std::cout << std::endl;
      }

      int j = 0;
      int offset2base = 0;
      for (int k=0; k < src.dim.stride.size(); ++k) {
        if (k==axis) {
          continue;
        }
        offset2base += dst_index[j] * src.dim.stride[k];
        j += 1;
      }

      if (axis == -1) { // sum all elements
        for (int b=0; b < src.batch_size(); ++b) {
          for (int i=0; i < src.dim.size(); ++i) {
            dst.data[b * ds] += src.data[i + b * ss];
          }
        }
      } else if (axis == src.dim.shape.size()) { // sum along batch
        for (int b=0; b < src.batch_size(); ++b) {
          int offset2 = offset2base;
          dst.data[offset1] += src.data[offset2 + b * ss];
        }
      } else { // sum along axis
        for (int b=0; b < src.batch_size(); ++b) {
          for (int i=0; i < src.dim.shape[axis]; ++i) {
            int offset2 = offset2base + i * src.dim.stride[axis];
            dst.data[offset1 + b * ds] += src.data[offset2 + b * ss];
          }
        }
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
  RNNPP_CHECK(dest.dim.batch_size == max_b, "Invalid batch_size");

  const float* ld = lhs.cdata();
  const float* rd = rhs.cdata();
  float* dd = dest.data;

  for (int i=0; i < dest.dim.size() * dest.dim.batch_size; ++i) {
    dd[i] = 0.;
  }

  for (int b=0; b < max_b; ++b) {
    for (int m=0; m < M; ++m) {
      for (int n=0; n < N; ++n) {
        int offset1 = m * dest.dim.stride[0] + n * dest.dim.stride[1];
        offset1 += dest.dim.size() * (b % dest.dim.batch_size);
        for (int k=0; k < K; ++k) {
          int offset2 = m * lhs.dim.stride[0] + k * lhs.dim.stride[1];
          offset2 += lhs.dim.size() * (b % lhs.dim.batch_size);
          int offset3 = k * rhs.dim.stride[0] + n * rhs.dim.stride[1];
          offset3 += rhs.dim.size() * (b % rhs.dim.batch_size);

          dd[offset1] += ld[offset2] * rd[offset3];
          
//          float l = ld[lhs.dim.size() * (b % lhs.dim.batch_size) + offset2];
//          float r = rd[rhs.dim.size() * (b % rhs.dim.batch_size) + offset3];
//          std::cout << "["<<m<<","<<n<<"]"<< " += [" <<m<<","<<k<<"]" << l << " * "\
//                    << "["<<k<<","<<n<<"]"<< r << " = " << dd[offset1] << std::endl;

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


// y_{i, T, k} = xs[N]_{i, j, k}
// T = sum_{t=1,...,N-1} xs[t].shape[axis] + j
//   = P + j
void _concatenate(std::vector<int> &dst_index, int pos, std::vector<Tensor> &xs,
    Tensor &dst, int axis) {
  if (pos == dst.dim.shape.size()-1) {
    int ds = dst.dim.size();

    for (dst_index[pos]=0; dst_index[pos] < dst.dim.shape[pos]; dst_index[pos] += 1) {

      int offset1 = 0;
      for (int k=0; k < dst_index.size(); ++k) {
        offset1 += dst_index[k] * dst.dim.stride[k];
      }

      if (axis == dst.dim.shape.size()) { // along batch
        int offset2 = 0;
        for (int k=0; k < dst_index.size(); ++k) {
          offset2 += dst_index[k] * xs[0].dim.stride[k];
        }

        int b_dst = 0;
        for (int N=0; N < xs.size(); ++N) {
          for (int b=0; b < xs[N].batch_size(); ++b) {
            int ss = xs[N].dim.size();
            dst.data[offset1 + b_dst * ds] = xs[N].data[offset2 + b * ss];
            b_dst += 1;
          }
        }
      } else { // along axis
        int T = dst_index[axis];
        int P = 0;
        int N = 0;
        for (; N < xs.size(); ++N) {
          float tmp = P + xs[N].dim.shape[axis];
          if (tmp > T) {
            break;
          }
          P = tmp;
        }
        int j = 0;
        for (; j < xs[N].dim.shape[axis]; ++j) {
          if (P + j == T) {
            break;
          }
        }

        int offset2 = 0;
        for (int k=0; k < dst_index.size(); ++k) {
          if (k == axis) {
            offset2 += j * xs[N].dim.stride[k];
          } else {
            offset2 += dst_index[k] * xs[N].dim.stride[k];
          }
        }

        for (int b=0; b < xs[N].batch_size(); ++b) {
          int ss = xs[N].dim.size();
          dst.data[offset1 + b * ds] = xs[N].data[offset2 + b * ss];
        }
      }
    }

  } else {
    for (dst_index[pos]=0; dst_index[pos] < dst.dim.shape[pos]; dst_index[pos] +=1) {
      _concatenate(dst_index, pos+1, xs, dst, axis);
    }
  }
}

void concatenate(std::vector<Tensor> &xs, Tensor &dst, int axis) {
  std::vector<int> dst_index(dst.dim.shape.size(), 0);
  _concatenate(dst_index, 0, xs, dst, axis);
}

} // namespace rnnpp
