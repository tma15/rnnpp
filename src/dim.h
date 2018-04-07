#ifndef RNNPP_DIM_H_
#define RNNPP_DIM_H_

#include <iostream>
#include <vector>

namespace rnnpp {

class Dim {

  public:
    Dim(){}

    ~Dim(){}

    Dim(std::vector<int> s): shape(s), batch_size(1) {
      set_stride(s);
    }

    Dim(std::vector<int> s, int b): shape(s), batch_size(b) {
      set_stride(s);
    }

    Dim(std::initializer_list<int> s): shape(s.begin(), s.end()), batch_size(1) {
      set_stride(shape);
    }

    Dim(std::initializer_list<int> s, int b): shape(s.begin(), s.end()), batch_size(b) {
      set_stride(shape);
    }

    friend std::ostream& operator<<(std::ostream &os, const Dim &d) {
      os << "(";
      for (int i=0; i < d.shape.size()-1; ++i) {
        os << d.shape[i] << ",";
      }
      os << d.shape.back() << ") " << d.batch_size;
      return os;
    };

    int operator[] (int i) const { return shape[i]; }

    bool operator== (const Dim &rhs) {
      if (shape.size() != rhs.shape.size()) {
        return false;
      }
      for (int i=0; i < shape.size(); ++i) {
        if (shape[i] != rhs.shape[i]) {
          return false;
        }
      }
      return true;
    }

    int size() const {
      int s = 1;
      for (int i=0; i < shape.size(); ++i) {
        s *= shape[i];
      }
      return s;
    }

    void set_stride(std::vector<int> s) {
      stride = std::vector<int>(s.size());
      stride.back() = 1;
      for (int i=0; i < s.size(); ++i) {
        stride[i] = 1;
        for (int j=i+1; j < s.size(); ++j) {
          stride[i] *= s[j];
        }
      }
    }

    std::vector<int> shape;

    /**
     * stride[ndim] = 1
     * stride[i] = prod_{j=i+1}^{ndim} shape[j]
     */
    std::vector<int> stride;

    int batch_size;


};

bool operator==(const Dim &lhs, const Dim &rhs);

} // namespace rnnpp

#endif // RNNPP_DIM_H_
