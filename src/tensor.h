#ifndef RNNPP_TENSOR_H_
#define RNNPP_TENSOR_H_

#include <math.h>

#include "dim.h"

namespace rnnpp {

namespace internal {

constexpr int adder(const std::vector<int> &shape, int k, int arg) {
  return shape[k] * arg;
}

template<typename ... Args> 
constexpr int adder(const std::vector<int> &shape, int k, int head, Args... tail) {
  return shape[k] * head + adder(shape, k + 1, tail...);
}

template<typename sub_t> 
struct Exp {
  inline const sub_t& self(void) const { return *static_cast<const sub_t*>(this); }
  inline sub_t* ptrself(void) { return static_cast<sub_t*>(this); }
};


struct SaveTo {
  inline static void save(float& x, float y) { x = y; }
};

struct AddTo {
  inline static void save(float& x, float y) { x += y; }
};

struct SubtractTo {
  inline static void save(float& x, float y) { x -= y; }
};

struct MultiplyTo {
  inline static void save(float& x, float y) { x *= y; }
};

struct DivideTo {
  inline static void save(float& x, float y) { x /= y; }
};

template<typename saver> 
struct ExpEngine {
  template<typename src_t, typename dst_t> 
  inline static void run(int size, Exp<dst_t> *dst, const Exp<src_t> &src) {
    dst_t dst_ = dst->self();
    src_t src_ = src.self();
    int max_b = src_.batch_size();

    for (int b=0; b < max_b; ++b) {
      for (int i=0; i < size; ++i) {
        saver::save(dst_.reval(i, b), src_.eval(i, b));
      }
    }
  }
};

struct Negative {
  inline static float map(float x) { return -x; }
};

struct Square {
  inline static float map(float x) { return x * x; }
};

struct Exponential {
  inline static float map(float x) { return exp(x); }
};

template<typename op, typename src_t> 
struct UnaryMapExp: public Exp< UnaryMapExp<op, src_t> > {
  const src_t& src;
  UnaryMapExp(const src_t &src): src(src) {}

  inline const float eval(int i, int b) const {
    return op::map(src.eval(i, b)); 
  }

  int batch_size() const { return src.batch_size(); }
};

template <typename op, typename src_t>
inline UnaryMapExp<op, src_t>
UnaryF(const Exp<src_t> &src) { return UnaryMapExp<op, src_t>(src.self()); }

template<typename src_t> 
inline UnaryMapExp<internal::Negative, src_t>
operator- (const Exp<src_t> &src) { return UnaryF<internal::Negative>(src); }

template<typename src_t> 
inline UnaryMapExp<internal::Square, src_t>
square(const Exp<src_t> &src) { return UnaryF<internal::Square>(src); }

template<typename src_t> 
inline UnaryMapExp<internal::Exponential, src_t>
exp(const Exp<src_t> &src) { return UnaryF<internal::Exponential>(src); }

struct MultTo {
  inline static float map(float &a, float b) { return a *= b; }
};

struct Mult {
  inline static float map(float a, float b) { return a * b; }
};

struct Add {
  inline static float map(float a, float b) { return a + b; }
};

struct Subtract {
  inline static float map(float a, float b) { return a - b; }
};

struct Division {
  inline static float map(float a, float b) { return a / b; }
};

template<typename op, typename lhs_t, typename rhs_t> 
struct BinaryMapExp: public Exp< BinaryMapExp<op, lhs_t, rhs_t> > {
  const lhs_t& lhs_;
  const rhs_t& rhs_;
  BinaryMapExp(const lhs_t &lhs, const rhs_t &rhs): lhs_(lhs), rhs_(rhs) {}

  inline const float eval(int i, int b) const {
    return op::map(lhs_.eval(i, b), rhs_.eval(i, b));
  }

  int batch_size() const { 
    return std::max(lhs_.self().batch_size(), rhs_.self().batch_size());
  }
};

template <typename op, typename lhs_t, typename rhs_t>
inline BinaryMapExp<op, lhs_t, rhs_t>
BinaryF(const Exp<lhs_t> &lhs, const Exp<rhs_t> &rhs) {
  return BinaryMapExp<op, lhs_t, rhs_t>(lhs.self(), rhs.self());
}

template<typename lhs_t, typename rhs_t> 
inline BinaryMapExp<internal::MultTo, lhs_t, rhs_t>
operator*= (const Exp<lhs_t> &lhs, const internal::Exp<rhs_t> &rhs) {
  return BinaryF<internal::MultTo>(lhs, rhs);
}


template<typename lhs_t, typename rhs_t> 
inline BinaryMapExp<internal::Mult, lhs_t, rhs_t>
operator* (const Exp<lhs_t> &lhs, const internal::Exp<rhs_t> &rhs) {
  return BinaryF<internal::Mult>(lhs, rhs);
}

template<typename lhs_t, typename rhs_t> 
inline BinaryMapExp<internal::Add, lhs_t, rhs_t>
operator+ (const Exp<lhs_t> &lhs, const internal::Exp<rhs_t> &rhs) {
  return BinaryF<internal::Add>(lhs, rhs);
}

template<typename lhs_t, typename rhs_t> 
inline BinaryMapExp<internal::Subtract, lhs_t, rhs_t>
operator- (const Exp<lhs_t> &lhs, const internal::Exp<rhs_t> &rhs) {
  return BinaryF<internal::Subtract>(lhs, rhs);
}

template<typename lhs_t, typename rhs_t> 
inline BinaryMapExp<internal::Division, lhs_t, rhs_t>
operator/ (const Exp<lhs_t> &lhs, const internal::Exp<rhs_t> &rhs) {
  return BinaryF<internal::Division>(lhs, rhs);
}


} // internal

class Scalar: public internal::Exp<Scalar> {
  public:
    Scalar(float v): data(v) {}

    ~Scalar() {}

    inline const float eval(int i, int b) const { return data; }

    int batch_size() const { return 1; }

  private:
    float data;
};


class Tensor: public internal::Exp<Tensor> {
  public:
    Tensor(): dim(Dim()), data(nullptr) {}

    Tensor(const Dim &d, const std::vector<float> &v): dim(d) {
      data = new float[v.size()];
      std::memcpy(data, v.data(), sizeof(float) * v.size());
    }

    ~Tensor() {}

    friend std::ostream& operator<<(std::ostream &os, const Tensor &t);

    template <typename ... Args>
    float& operator() (Args const & ... args) {
      int k = internal::adder(dim.stride, 0, args...);
      return data[k];
    }

    template <typename ... Args>
    float operator() (Args const & ... args) const {
      int k = internal::adder(dim.stride, 0, args...);
      return data[k];
    }

    template<typename src_t> 
    inline Tensor& operator=(const internal::Exp<src_t> &src_) {
      internal::ExpEngine<internal::SaveTo>::run(dim.size(), this->ptrself(), src_.self());
      return *this;
    }

    template<typename src_t> 
    inline Tensor& operator+=(const internal::Exp<src_t> &src_) {
      internal::ExpEngine<internal::AddTo>::run(dim.size(), this->ptrself(), src_.self());
      return *this;
    }

    template<typename src_t> 
    inline Tensor& operator-=(const internal::Exp<src_t> &src_) {
      internal::ExpEngine<internal::SubtractTo>::run(dim.size(), this->ptrself(), src_.self());
      return *this;
    }

    template<typename src_t> 
    inline Tensor& operator*=(const internal::Exp<src_t> &src_) {
      internal::ExpEngine<internal::MultiplyTo>::run(dim.size(), this->ptrself(), src_.self());
      return *this;
    }

    template<typename src_t> 
    inline Tensor& operator/=(const internal::Exp<src_t> &src_) {
      internal::ExpEngine<internal::DivideTo>::run(dim.size(), this->ptrself(), src_.self());
      return *this;
    }

    inline const float eval(int i, int b) const { 
      int skip = b * (dim.batch_size > 1) * dim.size();
      return data[i + skip];
    }

    inline float& reval(int i, int b) const { 
      int skip = b * (dim.batch_size > 1) * dim.size();
      return data[i + skip]; 
    }

    float* cdata() const { return data; }

    float* rdata() { return data; }

    Tensor transpose();

    int batch_size() const { return dim.batch_size; }

    float *data;
    Dim dim;
};


template<typename lhs_t> 
inline internal::BinaryMapExp<internal::Mult, lhs_t, Scalar>
operator* (const internal::Exp<lhs_t> &lhs, const Scalar &s) {
  return internal::BinaryF<internal::Mult>(lhs, s);
}

template<typename lhs_t> 
inline internal::BinaryMapExp<internal::Add, lhs_t, Scalar>
operator+ (const internal::Exp<lhs_t> &lhs, const Scalar &s) {
  return internal::BinaryF<internal::Add>(lhs, s);
}

template<typename lhs_t> 
inline internal::BinaryMapExp<internal::Subtract, lhs_t, Scalar>
operator- (const internal::Exp<lhs_t> &lhs, const Scalar &s) {
  return internal::BinaryF<internal::Subtract>(lhs, s);
}

template<typename lhs_t> 
inline internal::BinaryMapExp<internal::Division, lhs_t, Scalar>
operator/ (const internal::Exp<lhs_t> &lhs, const Scalar &s) {
  return internal::BinaryF<internal::Division>(lhs, s);
}


void matmul(const Tensor &lhs, const Tensor &rhs, Tensor &dest);

} // namespace rnnpp



#endif // RNNPP_TESOR_H_
