#include <iostream>

#include <gtest/gtest.h>

#include "../src/dim.h"
#include "../src/tensor.h"

using namespace rnnpp;


class TensorTest: public ::testing::Test {
  protected:
    void SetUp() {
      // Matrix
      std::vector<float> m1_data{0., 1.,
                                 2., 3.,
                                 4., 5.};
      m1 = Tensor(Dim({3, 2}), m1_data);

      std::vector<float> m2_data{1., 2.,
                                 3., 4.,
                                 5., 6.};
      m2 = Tensor(Dim({3, 2}), m2_data);

      std::vector<float> m3_data{2., 3.,
                                 4., 5.,
                                 6., 7.};
      m3 = Tensor(Dim({3, 2}), m3_data);

      m4.dim = Dim({2, 3});
      m4.data = new float[6];
      m4(0, 0) = 1.;
      m4(0, 1) = 2.;
      m4(0, 2) = 3.;
      m4(1, 0) = 4.;
      m4(1, 1) = 5.;
      m4(1, 2) = 6.;

      // Batched matrix
      std::vector<float> m_batch_data{0., 1.,
                                      2., 3.,
                                      4., 5.,
                                      6., 7.};
      m_batch = Tensor(Dim({2, 2}, 2), m_batch_data);

      std::vector<float> m_batch_data2{2., 3.,
                                       4., 5.,
                                       6., 7.,

                                       0., 1.,
                                       2., 3.,
                                       4., 5.};
      m_batch2 = Tensor(Dim({3, 2}, 2), m_batch_data2);

      std::vector<float> m_batch_data3{0., 1.,
                           2., 3.,
                           4., 5.,
                           2., 2.,
                           4., 6.,
                           5., 1.};
      m_batch3 = Tensor(Dim({3, 2}, 2), m_batch_data3);

      // 3D tensor
      std::vector<float> tensor3d_data{0., 1.,
                                       2., 3.,
                                       4., 5.,
                                       6., 7.};
      tensor3d = Tensor(Dim({2, 2, 2}), tensor3d_data);

      tensor4d.dim = Dim({2, 1, 3, 4});
      tensor4d.data = new float[24];
      for (int i=0; i < 24; ++i) tensor4d.data[i] = i;

      tensor5d.dim = Dim({2, 1, 5, 3, 4});
      tensor5d.data = new float[tensor5d.dim.size()];
      for (int i=0; i < tensor5d.dim.size(); ++i) tensor5d.data[i] = i;

      scalar = 3.f;

    };

    Tensor m1, m2, m3, m4;
    Tensor m_batch, m_batch2, m_batch3;
    Tensor tensor3d;
    Tensor tensor4d;
    Tensor tensor5d;

    float scalar;
};

TEST_F(TensorTest, Access) {
  EXPECT_EQ(m1(0, 0), 0.);
  EXPECT_EQ(m1(0, 1), 1.);
  EXPECT_EQ(m1(1, 0), 2.);
  EXPECT_EQ(m1(1, 1), 3.);
  EXPECT_EQ(m1(2, 0), 4.);
  EXPECT_EQ(m1(2, 1), 5.);
}

TEST_F(TensorTest, Transpose2D) {
  Tensor t = m1.transpose();
  EXPECT_EQ(t.dim[0], 2);
  EXPECT_EQ(t.dim[1], 3);
}

TEST_F(TensorTest, Transpose4D) {
  Tensor t = tensor4d.transpose();
  EXPECT_EQ(t.dim[0], 4);
  EXPECT_EQ(t.dim[1], 3);
  EXPECT_EQ(t.dim[2], 1);
  EXPECT_EQ(t.dim[3], 2);
}

TEST_F(TensorTest, Transpose5D) {
  Tensor t = tensor5d.transpose();
  EXPECT_EQ(t.dim[0], 4);
  EXPECT_EQ(t.dim[1], 3);
  EXPECT_EQ(t.dim[2], 5);
  EXPECT_EQ(t.dim[3], 1);
  EXPECT_EQ(t.dim[4], 2);
}

TEST_F(TensorTest, Sum) {
  Tensor dst;
  dst.dim = Dim({2, 2});
  dst.data = new float[dst.dim.size()];
  dst = Scalar(0.);
  sum(tensor3d, dst, 1);
//  array([[ 2,  4],
//         [10, 12]])
  EXPECT_EQ(dst(0, 0), 2.);
  EXPECT_EQ(dst(0, 1), 4.);
  EXPECT_EQ(dst(1, 0), 10.);
  EXPECT_EQ(dst(1, 1), 12.);
}

TEST_F(TensorTest, SumElem) {
  Tensor dst;
  dst.dim = Dim({1});
  dst.data = new float[dst.dim.size()];
  dst = Scalar(0.);
  sum(tensor3d, dst, -1);
  EXPECT_EQ(dst(0), 28.);
}

TEST_F(TensorTest, BatchSum) {
  Tensor dst;
  dst.dim = Dim({2}, 2);
  dst.data = new float[dst.dim.size()];
  dst = Scalar(0.);
  sum(m_batch, dst, 1);

  Tensor b1 = dst.batch_elem(0);
  EXPECT_EQ(b1(0), 1.);
  EXPECT_EQ(b1(1), 5.);

  Tensor b2 = dst.batch_elem(1);
  EXPECT_EQ(b2(0), 9.);
  EXPECT_EQ(b2(1), 13.);
}

TEST_F(TensorTest, BatchSumElem) {
  Tensor dst;
  dst.dim = Dim({1}, 2);
  dst.data = new float[dst.dim.size()];
  dst = Scalar(0.);
  sum(m_batch, dst, -1);

  Tensor b1 = dst.batch_elem(0);
  EXPECT_EQ(b1(0), 6.);
  Tensor b2 = dst.batch_elem(1);
  EXPECT_EQ(b2(0), 22.);
}

TEST_F(TensorTest, SumAlongBatch) {
  Tensor dst;
  dst.dim = Dim({2, 2}, 1);
  dst.data = new float[dst.dim.size()];
  dst = Scalar(0.);
  sum(m_batch, dst, 2);
  EXPECT_EQ(dst(0, 0), 4.);
  EXPECT_EQ(dst(0, 1), 6.);
  EXPECT_EQ(dst(1, 0), 8.);
  EXPECT_EQ(dst(1, 1), 10.);
}

TEST_F(TensorTest, ElementAdd) {
  Tensor res;
  res.dim = m1.dim;
  res.data = new float[res.dim.size()];
  res = m1 + m2;
  EXPECT_EQ(res(0, 0), 1.);
  EXPECT_EQ(res(0, 1), 3.);
  EXPECT_EQ(res(1, 0), 5.);
  EXPECT_EQ(res(1, 1), 7.);
  EXPECT_EQ(res(2, 0), 9.);
  EXPECT_EQ(res(2, 1), 11.);
}

TEST_F(TensorTest, ScalarMultiply) {
  Tensor res;
  res.dim = m1.dim;
  res.data = new float[res.dim.size()];
  res = Scalar(scalar) * m1;
  EXPECT_EQ(res(0, 0), m1(0, 0) * scalar);
  EXPECT_EQ(res(0, 1), m1(0, 1) * scalar);
  EXPECT_EQ(res(1, 0), m1(1, 0) * scalar);
  EXPECT_EQ(res(1, 1), m1(1, 1) * scalar);
  EXPECT_EQ(res(2, 0), m1(2, 0) * scalar);
  EXPECT_EQ(res(2, 1), m1(2, 1) * scalar);
}

TEST_F(TensorTest, BatchedElementAdd) {
  Tensor t;
  t.dim = Dim(m1.dim.shape, 2);
  t.data = new float[t.dim.size() * t.dim.batch_size];
  t = m1 + m_batch2;
  Tensor b0 = t.batch_elem(0);
  EXPECT_EQ(b0(0, 0), 2.);
  EXPECT_EQ(b0(0, 1), 4.);
  EXPECT_EQ(b0(1, 0), 6.);
  EXPECT_EQ(b0(1, 1), 8.);
  EXPECT_EQ(b0(2, 0), 10.);
  EXPECT_EQ(b0(2, 1), 12.);

  Tensor b1 = t.batch_elem(1);
  EXPECT_EQ(b1(0, 0), 0.);
  EXPECT_EQ(b1(0, 1), 2.);
  EXPECT_EQ(b1(1, 0), 4.);
  EXPECT_EQ(b1(1, 1), 6.);
  EXPECT_EQ(b1(2, 0), 8.);
  EXPECT_EQ(b1(2, 1), 10.);
}

TEST_F(TensorTest, Lengthy) {
  Tensor t;
  t.dim = m1.dim;
  t.data = new float[t.dim.size()];
  t = square((m1 - m2) * m3);
  EXPECT_EQ(t(0, 0), 4.);
  EXPECT_EQ(t(0, 1), 9.);
  EXPECT_EQ(t(1, 0), 16.);
  EXPECT_EQ(t(1, 1), 25.);
  EXPECT_EQ(t(2, 0), 36.);
  EXPECT_EQ(t(2, 1), 49.);
}

TEST_F(TensorTest, Matmul1) {
  Tensor res;
  res.dim = Dim({3, 3});
  res.data = new float[res.dim.size()];
  matmul(m1, m4, res);
  EXPECT_EQ(res(0, 0), 4.);
  EXPECT_EQ(res(0, 1), 5.);
  EXPECT_EQ(res(0, 2), 6.);
  EXPECT_EQ(res(1, 0), 14.);
  EXPECT_EQ(res(1, 1), 19.);
  EXPECT_EQ(res(1, 2), 24.);
  EXPECT_EQ(res(2, 0), 24.);
  EXPECT_EQ(res(2, 1), 33.);
  EXPECT_EQ(res(2, 2), 42.);
}

TEST_F(TensorTest, Matmul2) {
  Tensor res;
  res.dim = Dim({2, 2});
  res.data = new float[res.dim.size()];
  matmul(m1.transpose(), m4.transpose(), res);
  EXPECT_EQ(res(0, 0), 16.);
  EXPECT_EQ(res(0, 1), 34.);
  EXPECT_EQ(res(1, 0), 22.);
  EXPECT_EQ(res(1, 1), 49.);
}

TEST_F(TensorTest, BatchedMatmul) {
  Tensor res;
  res.dim = Dim({2, 2}, 2);
  res.data = new float[res.dim.size()];
  matmul(m_batch3.transpose(), m4.transpose(), res);
  Tensor ret1 = res.batch_elem(0);
  EXPECT_EQ(ret1(0, 0), 16.);
  EXPECT_EQ(ret1(0, 1), 34.);
  EXPECT_EQ(ret1(1, 0), 22.);
  EXPECT_EQ(ret1(1, 1), 49.);

  Tensor ret2 = res.batch_elem(1);
  EXPECT_EQ(ret2(0, 0), 25.);
  EXPECT_EQ(ret2(0, 1), 58.);
  EXPECT_EQ(ret2(1, 0), 17.);
  EXPECT_EQ(ret2(1, 1), 44.);
}

TEST_F(TensorTest, Concatenate) {
  Tensor res;
  res.dim = Dim({6, 2});
  res.data = new float[res.dim.size()];

  std::vector<Tensor> inputs = {m1, m2};
  concatenate(inputs, res, 0);
  EXPECT_EQ(res(0, 0), 0);
  EXPECT_EQ(res(0, 1), 1);
  EXPECT_EQ(res(1, 0), 2);
  EXPECT_EQ(res(1, 1), 3);
  EXPECT_EQ(res(2, 0), 4);
  EXPECT_EQ(res(2, 1), 5);
  EXPECT_EQ(res(3, 0), 1);
  EXPECT_EQ(res(3, 1), 2);
  EXPECT_EQ(res(4, 0), 3);
  EXPECT_EQ(res(4, 1), 4);
  EXPECT_EQ(res(5, 0), 5);
  EXPECT_EQ(res(5, 1), 6);
}

TEST_F(TensorTest, Concatenate2) {
  std::vector<Tensor> inputs = {m1, m2};
  Tensor res;
  res.dim = Dim({3, 4});
  res.data = new float[res.dim.size()];
  concatenate(inputs, res, 1);

  EXPECT_EQ(res(0, 0), 0);
  EXPECT_EQ(res(0, 1), 1);
  EXPECT_EQ(res(0, 2), 1);
  EXPECT_EQ(res(0, 3), 2);
  EXPECT_EQ(res(1, 0), 2);
  EXPECT_EQ(res(1, 1), 3);
  EXPECT_EQ(res(1, 2), 3);
  EXPECT_EQ(res(1, 3), 4);
  EXPECT_EQ(res(2, 0), 4);
  EXPECT_EQ(res(2, 1), 5);
  EXPECT_EQ(res(2, 2), 5);
  EXPECT_EQ(res(2, 3), 6);
}

TEST_F(TensorTest, ConcatenateAlongBatch) {
  Tensor res;
  res.dim = Dim({3, 2}, 3);
  res.data = new float[res.dim.size() * res.dim.batch_size];

  std::vector<Tensor> inputs = {m1, m_batch2};
  concatenate(inputs, res, 2);
  Tensor b0 = res.batch_elem(0);
  EXPECT_EQ(b0(0, 0), 0);
  EXPECT_EQ(b0(0, 1), 1);
  EXPECT_EQ(b0(1, 0), 2);
  EXPECT_EQ(b0(1, 1), 3);
  EXPECT_EQ(b0(2, 0), 4);
  EXPECT_EQ(b0(2, 1), 5);

  Tensor b1 = res.batch_elem(1);
  EXPECT_EQ(b1(0, 0), 2);
  EXPECT_EQ(b1(0, 1), 3);
  EXPECT_EQ(b1(1, 0), 4);
  EXPECT_EQ(b1(1, 1), 5);
  EXPECT_EQ(b1(2, 0), 6);
  EXPECT_EQ(b1(2, 1), 7);

  Tensor b2 = res.batch_elem(2);
  EXPECT_EQ(b2(0, 0), 0);
  EXPECT_EQ(b2(0, 1), 1);
  EXPECT_EQ(b2(1, 0), 2);
  EXPECT_EQ(b2(1, 1), 3);
  EXPECT_EQ(b2(2, 0), 4);
  EXPECT_EQ(b2(2, 1), 5);
}

TEST_F(TensorTest, Split) {
  Tensor res1, res2, res3;
  Dim d = Dim({1, 2}, 2);

  res1.dim = d;
  res1.data = new float[res1.dim.size(), res1.dim.batch_size];
  res2.dim = d;
  res2.data = new float[res2.dim.size(), res2.dim.batch_size];
  res3.dim = d;
  res3.data = new float[res3.dim.size(), res3.dim.batch_size];

  std::vector<Tensor> res = {res1, res2, res3};
  split(m_batch2, res, 0);

//  std::cout << "m_batch2: " << m_batch2.dim << "\n:" << m_batch2 << std::endl;
//  std::cout << "res1: " << res[0].dim << "\n:" << res[0] << std::endl;

  Tensor res1_b0 = res[0].batch_elem(0);
  EXPECT_EQ(res1_b0(0, 0), 2);
  EXPECT_EQ(res1_b0(0, 1), 3);

  Tensor res1_b1 = res[0].batch_elem(1);
  EXPECT_EQ(res1_b1(0, 0), 0);
  EXPECT_EQ(res1_b1(0, 1), 1);

  Tensor res2_b0 = res[1].batch_elem(0);
  EXPECT_EQ(res2_b0(0, 0), 4);
  EXPECT_EQ(res2_b0(0, 1), 5);

  Tensor res2_b1 = res[1].batch_elem(1);
  EXPECT_EQ(res2_b1(0, 0), 2);
  EXPECT_EQ(res2_b1(0, 1), 3);

  Tensor res3_b0 = res[2].batch_elem(0);
  EXPECT_EQ(res3_b0(0, 0), 6);
  EXPECT_EQ(res3_b0(0, 1), 7);

  Tensor res3_b1 = res[2].batch_elem(1);
  EXPECT_EQ(res3_b1(0, 0), 4);
  EXPECT_EQ(res3_b1(0, 1), 5);
}

TEST_F(TensorTest, SplitAlongBatch) {
  Tensor res1, res2;
  Dim d = Dim({2, 2});

  res1.dim = d;
  res1.data = new float[res1.dim.size()];
  res2.dim = d;
  res2.data = new float[res2.dim.size()];

  std::vector<Tensor> res = {res1, res2};
  split(m_batch, res, 2);

  std::cout << "res1\n:" << res[0] << std::endl;
  std::cout << "res2\n:" << res[1] << std::endl;

  EXPECT_EQ(res[0](0, 0), 0);
  EXPECT_EQ(res[0](0, 1), 1);
  EXPECT_EQ(res[0](1, 0), 2);
  EXPECT_EQ(res[0](1, 1), 3);

  EXPECT_EQ(res[1](0, 0), 4);
  EXPECT_EQ(res[1](0, 1), 5);
  EXPECT_EQ(res[1](1, 0), 6);
  EXPECT_EQ(res[1](1, 1), 7);

}
