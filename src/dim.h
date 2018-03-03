#ifndef RNNPP_DIM_H_
#define RNNPP_DIM_H_

#include <iostream>
#include <vector>

class Dim {
  public:
    Dim(){}
    ~Dim(){}

    Dim(std::vector<int> s): shape(s), batch_size(1) {}

    Dim(std::vector<int> s, int b): shape(s), batch_size(b) {}

    Dim(std::initializer_list<int> s): shape(s.begin(), s.end()), batch_size(1) {}

    Dim(std::initializer_list<int> s, int b): shape(s.begin(), s.end()), batch_size(b) {
    }

    friend std::ostream& operator<<(std::ostream &os, const Dim &d) {
      os << "(";
      for (int i=0; i < d.shape.size()-1; ++i) {
        os << d.shape[i] << ",";
      }
      os << d.shape[d.shape.size()-1] << ")";
      return os;
    };

    int operator[] (int i) const { return shape[i]; }

    int size() const {
      int s = 1;
      for (int i=0; i < shape.size(); ++i) {
        s *= shape[i];
      }
      return s;
    }

    std::vector<int> shape;

    int batch_size;

};

#endif // RNNPP_DIM_H_
