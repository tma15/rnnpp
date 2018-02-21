#ifndef RNNPP_DIM_H_
#define RNNPP_DIM_H_

#include <iostream>

class Dim {
  public:
    Dim(){}
    ~Dim(){}

    Dim(std::initializer_list<int> s): shape(s.begin(), s.end()), batch_size(1) {}

    Dim(std::initializer_list<int> s, int b): shape(s.begin(), s.end()), batch_size(b) {}

    std::vector<int> shape;
    int batch_size;

    int size() const {
      int s = 1;
      for (int i=0; i < shape.size(); ++i) {
        s *= shape[i];
      }
      return s;
    }

    void print_shape() {
      std::cout << "(";
      for (int i=0; i < shape.size()-1; ++i) {
        std::cout << shape[i] << ",";
      }
      std::cout << shape[shape.size()-1] << ")" << std::endl;
    }

};

#endif // RNNPP_DIM_H_
