#include <iostream>

#include <gtest/gtest.h>

#include "../src/dim.h"
#include "../src/tensor.h"

using namespace rnnpp;


class TensorAccessTest: public ::testing::Test {
  protected:
    void SetUp() {
      std::vector<float> v{0., 1.,
                           2., 3.,
                           4., 5.};
      t1 = Tensor(Dim({3, 2}), v);
    };
    Tensor t1;
};

TEST_F(TensorAccessTest, Access) {
  EXPECT_EQ(t1.data[0], 0.);
  EXPECT_EQ(t1.data[1], 1.);
  EXPECT_EQ(t1.data[2], 2.);
  EXPECT_EQ(t1.data[3], 3.);
  EXPECT_EQ(t1.data[4], 4.);
  EXPECT_EQ(t1.data[5], 5.);
}

TEST_F(TensorAccessTest, Access2) {
  EXPECT_EQ(t1(0, 0), 0.);
  EXPECT_EQ(t1(0, 1), 1.);
  EXPECT_EQ(t1(1, 0), 2.);
  EXPECT_EQ(t1(1, 1), 3.);
  EXPECT_EQ(t1(2, 0), 4.);
  EXPECT_EQ(t1(2, 1), 5.);
}


TEST_F(TensorAccessTest, Transpose0) {
  Tensor t2 = t1.transpose();
  EXPECT_EQ(t2.dim[0], 2);
  EXPECT_EQ(t2.dim[1], 3);
}


TEST_F(TensorAccessTest, Transpose1) {
  Tensor t1;
  t1.dim = Dim({2, 1, 3, 4});
  t1.data = new float[24];
  for (int i=0; i < 24; ++i) t1.data[i] = i;

  Tensor t2 = t1.transpose();
  EXPECT_EQ(t2.dim[0], 4);
  EXPECT_EQ(t2.dim[1], 3);
  EXPECT_EQ(t2.dim[2], 1);
  EXPECT_EQ(t2.dim[3], 2);
}

TEST_F(TensorAccessTest, Transpose2) {
  Tensor t1;
  t1.dim = Dim({2, 1, 5, 3, 4});
  Tensor t2 = t1.transpose();
  EXPECT_EQ(t2.dim[0], 4);
  EXPECT_EQ(t2.dim[1], 3);
  EXPECT_EQ(t2.dim[2], 5);
  EXPECT_EQ(t2.dim[3], 1);
  EXPECT_EQ(t2.dim[4], 2);
}

class TensorElementwiseTest: public ::testing::Test {
  protected:
    void SetUp() {
      std::vector<float> v1{0., 1.,
                            2., 3.,
                            4., 5.};
      t1 = Tensor(Dim({3, 2}), v1);

      std::vector<float> v2{1., 2.,
                            3., 4.,
                            5., 6.};
      t2 = Tensor(Dim({3, 2}), v2);

      std::vector<float> v3{2., 3.,
                            4., 5.,
                            6., 7.};
      t3 = Tensor(Dim({3, 2}), v3);

      std::vector<float> v4{2., 3.,
                            4., 5.,
                            6., 7.,

                            0., 1.,
                            2., 3.,
                            4., 5.};
      t4 = Tensor(Dim({3, 2}, 2), v4);

      scalar = 3.f;
    }

    Tensor t1;
    Tensor t2;
    Tensor t3;
    Tensor t4;
    float scalar;
};

TEST_F(TensorElementwiseTest, Sum) {
  std::vector<float> v1{0., 1.,
                        2., 3.,
                        4., 5.,
                        6., 7.};
  Tensor t = Tensor(Dim({2, 2, 2}), v1);

//  std::cout << t << std::endl;
  Tensor dst;
  dst.dim = Dim({2, 2});
  dst.data = new float[dst.dim.size()];
  sum(t, dst, 1);

//  array([[ 2,  4],
//         [10, 12]])
  EXPECT_EQ(dst(0, 0), 2.);
  EXPECT_EQ(dst(0, 1), 4.);
  EXPECT_EQ(dst(1, 0), 10.);
  EXPECT_EQ(dst(1, 1), 12.);
//  std::cout << "ret:" << dst.dim << std::endl;
//  std::cout << dst << std::endl;
}

TEST_F(TensorElementwiseTest, BatchSum) {
  std::vector<float> v1{0., 1.,
                        2., 3.,
                        4., 5.,
                        6., 7.};
  Tensor t = Tensor(Dim({2, 2}, 2), v1);

  std::cout << "src" << std::endl;
  std::cout << t << std::endl;

//  std::cout << t << std::endl;
  Tensor dst;
  dst.dim = Dim({2}, 2);
  dst.data = new float[dst.dim.size()];
  sum(t, dst, 1);

//  array([[ 2,  4],
//         [10, 12]])
//  EXPECT_EQ(dst(0, 0), 2.);
//  EXPECT_EQ(dst(0, 1), 4.);
//  EXPECT_EQ(dst(1, 0), 10.);
//  EXPECT_EQ(dst(1, 1), 12.);
  std::cout << "ret:" << dst.dim << std::endl;
  std::cout << dst << std::endl;
}

TEST_F(TensorElementwiseTest, ElementAdd) {
  Tensor t;
  t.dim = t1.dim;
  t.data = new float[t.dim.size()];
  t = t1 + t2;
  EXPECT_EQ(t(0, 0), 1.);
  EXPECT_EQ(t(0, 1), 3.);
  EXPECT_EQ(t(1, 0), 5.);
  EXPECT_EQ(t(1, 1), 7.);
  EXPECT_EQ(t(2, 0), 9.);
  EXPECT_EQ(t(2, 1), 11.);

  Tensor t3;
  t3.dim = t1.dim;
  t3.data = new float[t3.dim.size()];
  t3 = Scalar(scalar) * t1;
//  t3 = t1 * Scalar(scalar);
  EXPECT_EQ(t3(0, 0), t1(0, 0) * scalar);
  EXPECT_EQ(t3(0, 1), t1(0, 1) * scalar);
  EXPECT_EQ(t3(1, 0), t1(1, 0) * scalar);
  EXPECT_EQ(t3(1, 1), t1(1, 1) * scalar);
  EXPECT_EQ(t3(2, 0), t1(2, 0) * scalar);
  EXPECT_EQ(t3(2, 1), t1(2, 1) * scalar);

}

TEST_F(TensorElementwiseTest, BatchedElementAdd) {
  Tensor t;
  t.dim = Dim(t1.dim.shape, 2);
  t.data = new float[t.dim.size() * 2];
  t = t1 + t4;
  std::cout << "t:" << t << std::endl;
//  EXPECT_EQ(t(0, 0), 1.);
//  EXPECT_EQ(t(0, 1), 3.);
//  EXPECT_EQ(t(1, 0), 5.);
//  EXPECT_EQ(t(1, 1), 7.);
//  EXPECT_EQ(t(2, 0), 9.);
//  EXPECT_EQ(t(2, 1), 11.);
  std::cout << "fin: " << std::endl;
}


TEST_F(TensorElementwiseTest, Lengthy) {
  Tensor t;
  t.dim = t1.dim;
  t.data = new float[t.dim.size()];
  t = square((t1 - t2) * t3);
//  std::cout << t << std::endl;
  EXPECT_EQ(t(0, 0), 4.);
  EXPECT_EQ(t(0, 1), 9.);
  EXPECT_EQ(t(1, 0), 16.);
  EXPECT_EQ(t(1, 1), 25.);
  EXPECT_EQ(t(2, 0), 36.);
  EXPECT_EQ(t(2, 1), 49.);
}


class TensorMatmulTest: public ::testing::Test {
  protected:
    void SetUp() {
      t1.dim = Dim({3, 2});
      t1.data = new float[6];
      t1.data[0] = 0.;
      t1.data[1] = 1.;
      t1.data[2] = 2.;
      t1.data[3] = 3.;
      t1.data[4] = 4.;
      t1.data[5] = 5.;

      std::vector<float> v{0., 1.,
                           2., 3.,
                           4., 5.,
                           0., 1.,
                           2., 3.,
                           4., 5.};
      t1_batched = Tensor(Dim({3, 2}, 2), v);

      t2.dim = Dim({2, 3});
      t2.data = new float[6];
      t2.data[0] = 1.;
      t2.data[1] = 2.;
      t2.data[2] = 3.;
      t2.data[3] = 4.;
      t2.data[4] = 5.;
      t2.data[5] = 6.;

      t3.dim = Dim({3, 3});
      t3.data = new float[9];

      t4.dim = Dim({2, 2});
      t4.data = new float[4];
    }

    Tensor t1;
    Tensor t1_batched;
    Tensor t2;
    Tensor t3;
    Tensor t4;
};

TEST_F(TensorMatmulTest, Matmul1) {
  matmul(t1, t2, t3);
//  array([[ 4,  5,  6],
//         [14, 19, 24],
//         [24, 33, 42]])
//  std::cout << t3 << std::endl;
  EXPECT_EQ(t3.data[0], 4.);
  EXPECT_EQ(t3.data[1], 5.);
  EXPECT_EQ(t3.data[2], 6.);
  EXPECT_EQ(t3.data[3], 14.);
  EXPECT_EQ(t3.data[4], 19.);
  EXPECT_EQ(t3.data[5], 24.);
  EXPECT_EQ(t3.data[6], 24.);
  EXPECT_EQ(t3.data[7], 33.);
  EXPECT_EQ(t3.data[8], 42.);
}


TEST_F(TensorMatmulTest, Matmul2) {
//  std::cout << t1 << std::endl;
//  std::cout << t2 << std::endl;
//  std::cout << t1.transpose() << std::endl;
//  std::cout << t2.transpose() << std::endl;
  matmul(t1.transpose(), t2.transpose(), t4);
  EXPECT_EQ(t4(0, 0), 16.);
  EXPECT_EQ(t4(0, 1), 34.);
  EXPECT_EQ(t4(1, 0), 22.);
  EXPECT_EQ(t4(1, 1), 49.);
}

TEST_F(TensorMatmulTest, Matmul3) {
  std::vector<float> v{-0.0961903, 0.154988, -0.0179993,
                       -0.00139399, -0.0737388, 0.0992229};
  Tensor ta = Tensor(Dim({3, 2}, 1), v);

  std::vector<float> v2{-0.181541,  0.0185337,
                        -0.0493364, 0.136482,
                        -0.01556, -0.0303789};
  Tensor tb = Tensor(Dim({3, 2}, 1), v2);

  std::vector<float> v3{1., 1.,
                        1., 1.};
  Tensor tc = Tensor(Dim({2, 2}, 1), v3);

  Tensor t4;
  t4.dim  = Dim({2, 3}, 1);
  t4.data = new float[6];

//  std::cout << tc.dim << " " << tb.dim << std::endl;
//  std::cout << "tc:\n" << tc << std::endl;
//  std::cout << "tb\n" << tb << std::endl;
//  std::cout << "tb.T\n" << tb.transpose() << std::endl;
//  matmul(tc, tb.transpose(), t4);
//  std::cout << "t4" << std::endl;
//  std::cout << t4 << std::endl;
//  np.matmul(dEdy, b.transpose())
//  array([[-0.045059 ,  0.0029737, -0.0797153],
//         [-0.045059 ,  0.0029737, -0.0797153]])
//
//  std::cout << t3 << std::endl;
//  EXPECT_EQ(t3.data[0], 4.);
//  EXPECT_EQ(t3.data[1], 5.);
//  EXPECT_EQ(t3.data[2], 6.);
//  EXPECT_EQ(t3.data[3], 14.);
//  EXPECT_EQ(t3.data[4], 19.);
//  EXPECT_EQ(t3.data[5], 24.);
//  EXPECT_EQ(t3.data[6], 24.);
//  EXPECT_EQ(t3.data[7], 33.);
//  EXPECT_EQ(t3.data[8], 42.);
}


//TEST_F(TensorMatmulTest, BatchedMatmul3) {
//  matmul(t1_batched.transpose(), t2.transpose(), t4);
//  array([[13, 16],
//        [40, 52]])
//  std::cout << t4 << std::endl;
//  EXPECT_EQ(t4.data[0], 26.);
//  EXPECT_EQ(t4.data[1], 32.);
//  EXPECT_EQ(t4.data[2], 80.);
//  EXPECT_EQ(t4.data[3], 104.);
//}

