/*******************************************************************************
* Copyright 2020 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions
* and limitations under the License.
*
*
* SPDX-License-Identifier: Apache-2.0
*******************************************************************************/

#include <CL/sycl.hpp>

#include "cpu_common.hpp"

namespace onemkl {
namespace blis {

void gemm(cl::sycl::queue &queue, transpose transa, transpose transb, int64_t m, int64_t n,
          int64_t k, float alpha, cl::sycl::buffer<float, 1> &a, int64_t lda,
          cl::sycl::buffer<float, 1> &b, int64_t ldb, float beta, cl::sycl::buffer<float, 1> &c,
          int64_t ldc) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char transa_ = *fortran_char(transa);
        const char transb_ = *fortran_char(transb);
        auto accessor_a    = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b    = b.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_c    = c.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class blis_kernel_sgemm>(cgh, [=]() {
            ::sgemm((const char *)&transa_, (const char *)&transb_, (const gint_t *)&m,
                    (const gint_t *)&n, (const gint_t *)&k, (const float *)&alpha,
                    accessor_a.get_pointer(), (const gint_t *)&lda, accessor_b.get_pointer(),
                    (const gint_t *)&ldb, (const float *)&beta, accessor_c.get_pointer(),
                    (const gint_t *)&ldc);
        });
    });
}

void gemm(cl::sycl::queue &queue, transpose transa, transpose transb, int64_t m, int64_t n,
          int64_t k, double alpha, cl::sycl::buffer<double, 1> &a, int64_t lda,
          cl::sycl::buffer<double, 1> &b, int64_t ldb, double beta, cl::sycl::buffer<double, 1> &c,
          int64_t ldc) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char transa_ = *fortran_char(transa);
        const char transb_ = *fortran_char(transb);
        auto accessor_a    = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b    = b.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_c    = c.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class blis_kernel_dgemm>(cgh, [=]() {
            ::dgemm((const char *)&transa_, (const char *)&transb_, (const gint_t *)&m,
                    (const gint_t *)&n, (const gint_t *)&k, (const double *)&alpha,
                    accessor_a.get_pointer(), (const gint_t *)&lda, accessor_b.get_pointer(),
                    (const gint_t *)&ldb, (const double *)&beta, accessor_c.get_pointer(),
                    (const gint_t *)&ldc);
        });
    });
}

void gemm(cl::sycl::queue &queue, transpose transa, transpose transb, int64_t m, int64_t n,
          int64_t k, std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
          int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &b, int64_t ldb,
          std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &c, int64_t ldc) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char transa_ = *fortran_char(transa);
        const char transb_ = *fortran_char(transb);
        float alpha_real = alpha.real(), alpha_imag = alpha.imag();
        float beta_real = beta.real(), beta_imag = beta.imag();
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b = b.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_c = c.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class blis_kernel_cgemm>(cgh, [=]() {
            BLIS_Complex8 alpha_ = { alpha_real, alpha_imag };
            BLIS_Complex8 beta_  = { beta_real, beta_imag };
            ::cgemm((const char *)&transa_, (const char *)&transb_, (const gint_t *)&m,
                    (const gint_t *)&n, (const gint_t *)&k, (const BLIS_Complex8 *)&alpha_,
                    accessor_a.get_pointer(), (const gint_t *)&lda, accessor_b.get_pointer(),
                    (const gint_t *)&ldb, (const BLIS_Complex8 *)&beta_, accessor_c.get_pointer(),
                    (const gint_t *)&ldc);
        });
    });
}

void gemm(cl::sycl::queue &queue, transpose transa, transpose transb, int64_t m, int64_t n,
          int64_t k, std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
          int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &b, int64_t ldb,
          std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c, int64_t ldc) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char transa_ = *fortran_char(transa);
        const char transb_ = *fortran_char(transb);
        double alpha_real = alpha.real(), alpha_imag = alpha.imag();
        double beta_real = beta.real(), beta_imag = beta.imag();
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b = b.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_c = c.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class blis_kernel_zgemm>(cgh, [=]() {
            BLIS_Complex16 alpha_ = { alpha_real, alpha_imag };
            BLIS_Complex16 beta_  = { beta_real, beta_imag };
            ::zgemm((const char *)&transa_, (const char *)&transb_, (const gint_t *)&m,
                    (const gint_t *)&n, (const gint_t *)&k, (const BLIS_Complex16 *)&alpha_,
                    accessor_a.get_pointer(), (const gint_t *)&lda, accessor_b.get_pointer(),
                    (const gint_t *)&ldb, (const BLIS_Complex16 *)&beta_, accessor_c.get_pointer(),
                    (const gint_t *)&ldc);
        });
    });
}

void hemm(cl::sycl::queue &queue, side left_right, uplo upper_lower, int64_t m, int64_t n,
          std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &b, int64_t ldb, std::complex<float> beta,
          cl::sycl::buffer<std::complex<float>, 1> &c, int64_t ldc) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char left_right_  = *fortran_char(left_right);
        const char upper_lower_ = *fortran_char(upper_lower);
        float alpha_real = alpha.real(), alpha_imag = alpha.imag();
        float beta_real = beta.real(), beta_imag = beta.imag();
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b = b.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_c = c.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class blis_kernel_chemm>(cgh, [=]() {
            BLIS_Complex8 alpha_ = { alpha_real, alpha_imag };
            BLIS_Complex8 beta_  = { beta_real, beta_imag };
            ::chemm((const char *)&left_right_, (const char *)&upper_lower_, (const gint_t *)&m,
                    (const gint_t *)&n, (const BLIS_Complex8 *)&alpha_, accessor_a.get_pointer(),
                    (const gint_t *)&lda, accessor_b.get_pointer(), (const gint_t *)&ldb,
                    (const BLIS_Complex8 *)&beta_, accessor_c.get_pointer(), (const gint_t *)&ldc);
        });
    });
}

void hemm(cl::sycl::queue &queue, side left_right, uplo upper_lower, int64_t m, int64_t n,
          std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &b, int64_t ldb, std::complex<double> beta,
          cl::sycl::buffer<std::complex<double>, 1> &c, int64_t ldc) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char left_right_  = *fortran_char(left_right);
        const char upper_lower_ = *fortran_char(upper_lower);
        double alpha_real = alpha.real(), alpha_imag = alpha.imag();
        double beta_real = beta.real(), beta_imag = beta.imag();
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b = b.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_c = c.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class blis_kernel_zhemm>(cgh, [=]() {
            BLIS_Complex16 alpha_ = { alpha_real, alpha_imag };
            BLIS_Complex16 beta_  = { beta_real, beta_imag };
            ::zhemm((const char *)&left_right_, (const char *)&upper_lower_, (const gint_t *)&m,
                    (const gint_t *)&n, (const BLIS_Complex16 *)&alpha_, accessor_a.get_pointer(),
                    (const gint_t *)&lda, accessor_b.get_pointer(), (const gint_t *)&ldb,
                    (const BLIS_Complex16 *)&beta_, accessor_c.get_pointer(), (const gint_t *)&ldc);
        });
    });
}

void herk(cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
          float alpha, cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda, float beta,
          cl::sycl::buffer<std::complex<float>, 1> &c, int64_t ldc) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char upper_lower_ = *fortran_char(upper_lower);
        const char trans_       = *fortran_char(trans);
        auto accessor_a         = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_c         = c.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class blis_kernel_cherk>(cgh, [=]() {
            ::cherk((const char *)&upper_lower_, (const char *)&trans_, (const gint_t *)&n,
                    (const gint_t *)&k, (const float *)&alpha, accessor_a.get_pointer(),
                    (const gint_t *)&lda, (const float *)&beta, accessor_c.get_pointer(),
                    (const gint_t *)&ldc);
        });
    });
}

void herk(cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
          double alpha, cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda, double beta,
          cl::sycl::buffer<std::complex<double>, 1> &c, int64_t ldc) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char upper_lower_ = *fortran_char(upper_lower);
        const char trans_       = *fortran_char(trans);
        auto accessor_a         = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_c         = c.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class blis_kernel_zherk>(cgh, [=]() {
            ::zherk((const char *)&upper_lower_, (const char *)&trans_, (const gint_t *)&n,
                    (const gint_t *)&k, (const double *)&alpha, accessor_a.get_pointer(),
                    (const gint_t *)&lda, (const double *)&beta, accessor_c.get_pointer(),
                    (const gint_t *)&ldc);
        });
    });
}

void her2k(cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
           std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
           cl::sycl::buffer<std::complex<float>, 1> &b, int64_t ldb, float beta,
           cl::sycl::buffer<std::complex<float>, 1> &c, int64_t ldc) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char upper_lower_ = *fortran_char(upper_lower);
        const char trans_       = *fortran_char(trans);
        float alpha_real = alpha.real(), alpha_imag = alpha.imag();
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b = b.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_c = c.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class blis_kernel_cher2k>(cgh, [=]() {
            BLIS_Complex8 alpha_ = { alpha_real, alpha_imag };
            ::cher2k((const char *)&upper_lower_, (const char *)&trans_, (const gint_t *)&n,
                     (const gint_t *)&k, (const BLIS_Complex8 *)&alpha_, accessor_a.get_pointer(),
                     (const gint_t *)&lda, accessor_b.get_pointer(), (const gint_t *)&ldb,
                     (const float *)&beta, accessor_c.get_pointer(), (const gint_t *)&ldc);
        });
    });
}

void her2k(cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
           std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
           cl::sycl::buffer<std::complex<double>, 1> &b, int64_t ldb, double beta,
           cl::sycl::buffer<std::complex<double>, 1> &c, int64_t ldc) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char upper_lower_ = *fortran_char(upper_lower);
        const char trans_       = *fortran_char(trans);
        double alpha_real = alpha.real(), alpha_imag = alpha.imag();
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b = b.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_c = c.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class blis_kernel_zher2k>(cgh, [=]() {
            BLIS_Complex16 alpha_ = { alpha_real, alpha_imag };
            ::zher2k((const char *)&upper_lower_, (const char *)&trans_, (const gint_t *)&n,
                     (const gint_t *)&k, (const BLIS_Complex16 *)&alpha_, accessor_a.get_pointer(),
                     (const gint_t *)&lda, accessor_b.get_pointer(), (const gint_t *)&ldb,
                     (const double *)&beta, accessor_c.get_pointer(), (const gint_t *)&ldc);
        });
    });
}

void symm(cl::sycl::queue &queue, side left_right, uplo upper_lower, int64_t m, int64_t n,
          float alpha, cl::sycl::buffer<float, 1> &a, int64_t lda, cl::sycl::buffer<float, 1> &b,
          int64_t ldb, float beta, cl::sycl::buffer<float, 1> &c, int64_t ldc) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char left_right_  = *fortran_char(left_right);
        const char upper_lower_ = *fortran_char(upper_lower);
        auto accessor_a         = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b         = b.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_c         = c.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class blis_kernel_ssymm>(cgh, [=]() {
            ::ssymm((const char *)&left_right_, (const char *)&upper_lower_, (const gint_t *)&m,
                    (const gint_t *)&n, (const float *)&alpha, accessor_a.get_pointer(),
                    (const gint_t *)&lda, accessor_b.get_pointer(), (const gint_t *)&ldb,
                    (const float *)&beta, accessor_c.get_pointer(), (const gint_t *)&ldc);
        });
    });
}

void symm(cl::sycl::queue &queue, side left_right, uplo upper_lower, int64_t m, int64_t n,
          double alpha, cl::sycl::buffer<double, 1> &a, int64_t lda, cl::sycl::buffer<double, 1> &b,
          int64_t ldb, double beta, cl::sycl::buffer<double, 1> &c, int64_t ldc) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char left_right_  = *fortran_char(left_right);
        const char upper_lower_ = *fortran_char(upper_lower);
        auto accessor_a         = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b         = b.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_c         = c.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class blis_kernel_dsymm>(cgh, [=]() {
            ::dsymm((const char *)&left_right_, (const char *)&upper_lower_, (const gint_t *)&m,
                    (const gint_t *)&n, (const double *)&alpha, accessor_a.get_pointer(),
                    (const gint_t *)&lda, accessor_b.get_pointer(), (const gint_t *)&ldb,
                    (const double *)&beta, accessor_c.get_pointer(), (const gint_t *)&ldc);
        });
    });
}

void symm(cl::sycl::queue &queue, side left_right, uplo upper_lower, int64_t m, int64_t n,
          std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &b, int64_t ldb, std::complex<float> beta,
          cl::sycl::buffer<std::complex<float>, 1> &c, int64_t ldc) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char left_right_  = *fortran_char(left_right);
        const char upper_lower_ = *fortran_char(upper_lower);
        float alpha_real = alpha.real(), alpha_imag = alpha.imag();
        float beta_real = beta.real(), beta_imag = beta.imag();
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b = b.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_c = c.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class blis_kernel_csymm>(cgh, [=]() {
            BLIS_Complex8 alpha_ = { alpha_real, alpha_imag };
            BLIS_Complex8 beta_  = { beta_real, beta_imag };
            ::csymm((const char *)&left_right_, (const char *)&upper_lower_, (const gint_t *)&m,
                    (const gint_t *)&n, (const BLIS_Complex8 *)&alpha_, accessor_a.get_pointer(),
                    (const gint_t *)&lda, accessor_b.get_pointer(), (const gint_t *)&ldb,
                    (const BLIS_Complex8 *)&beta_, accessor_c.get_pointer(), (const gint_t *)&ldc);
        });
    });
}

void symm(cl::sycl::queue &queue, side left_right, uplo upper_lower, int64_t m, int64_t n,
          std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &b, int64_t ldb, std::complex<double> beta,
          cl::sycl::buffer<std::complex<double>, 1> &c, int64_t ldc) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char left_right_  = *fortran_char(left_right);
        const char upper_lower_ = *fortran_char(upper_lower);
        double alpha_real = alpha.real(), alpha_imag = alpha.imag();
        double beta_real = beta.real(), beta_imag = beta.imag();
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b = b.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_c = c.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class blis_kernel_zsymm>(cgh, [=]() {
            BLIS_Complex16 alpha_ = { alpha_real, alpha_imag };
            BLIS_Complex16 beta_  = { beta_real, beta_imag };
            ::zsymm((const char *)&left_right_, (const char *)&upper_lower_, (const gint_t *)&m,
                    (const gint_t *)&n, (const BLIS_Complex16 *)&alpha_, accessor_a.get_pointer(),
                    (const gint_t *)&lda, accessor_b.get_pointer(), (const gint_t *)&ldb,
                    (const BLIS_Complex16 *)&beta_, accessor_c.get_pointer(), (const gint_t *)&ldc);
        });
    });
}

void syrk(cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
          float alpha, cl::sycl::buffer<float, 1> &a, int64_t lda, float beta,
          cl::sycl::buffer<float, 1> &c, int64_t ldc) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char upper_lower_ = *fortran_char(upper_lower);
        const char trans_       = *fortran_char(trans);
        auto accessor_a         = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_c         = c.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class blis_kernel_ssyrk>(cgh, [=]() {
            ::ssyrk((const char *)&upper_lower_, (const char *)&trans_, (const gint_t *)&n,
                    (const gint_t *)&k, (const float *)&alpha, accessor_a.get_pointer(),
                    (const gint_t *)&lda, (const float *)&beta, accessor_c.get_pointer(),
                    (const gint_t *)&ldc);
        });
    });
}

void syrk(cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
          double alpha, cl::sycl::buffer<double, 1> &a, int64_t lda, double beta,
          cl::sycl::buffer<double, 1> &c, int64_t ldc) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char upper_lower_ = *fortran_char(upper_lower);
        const char trans_       = *fortran_char(trans);
        auto accessor_a         = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_c         = c.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class blis_kernel_dsyrk>(cgh, [=]() {
            ::dsyrk((const char *)&upper_lower_, (const char *)&trans_, (const gint_t *)&n,
                    (const gint_t *)&k, (const double *)&alpha, accessor_a.get_pointer(),
                    (const gint_t *)&lda, (const double *)&beta, accessor_c.get_pointer(),
                    (const gint_t *)&ldc);
        });
    });
}

void syrk(cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
          std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
          std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &c, int64_t ldc) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char upper_lower_ = *fortran_char(upper_lower);
        const char trans_       = *fortran_char(trans);
        float alpha_real = alpha.real(), alpha_imag = alpha.imag();
        float beta_real = beta.real(), beta_imag = beta.imag();
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_c = c.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class blis_kernel_csyrk>(cgh, [=]() {
            BLIS_Complex8 alpha_ = { alpha_real, alpha_imag };
            BLIS_Complex8 beta_  = { beta_real, beta_imag };
            ::csyrk((const char *)&upper_lower_, (const char *)&trans_, (const gint_t *)&n,
                    (const gint_t *)&k, (const BLIS_Complex8 *)&alpha_, accessor_a.get_pointer(),
                    (const gint_t *)&lda, (const BLIS_Complex8 *)&beta_, accessor_c.get_pointer(),
                    (const gint_t *)&ldc);
        });
    });
}

void syrk(cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
          std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
          std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c, int64_t ldc) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char upper_lower_ = *fortran_char(upper_lower);
        const char trans_       = *fortran_char(trans);
        double alpha_real = alpha.real(), alpha_imag = alpha.imag();
        double beta_real = beta.real(), beta_imag = beta.imag();
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_c = c.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class blis_kernel_zsyrk>(cgh, [=]() {
            BLIS_Complex16 alpha_ = { alpha_real, alpha_imag };
            BLIS_Complex16 beta_  = { beta_real, beta_imag };
            ::zsyrk((const char *)&upper_lower_, (const char *)&trans_, (const gint_t *)&n,
                    (const gint_t *)&k, (const BLIS_Complex16 *)&alpha_, accessor_a.get_pointer(),
                    (const gint_t *)&lda, (const BLIS_Complex16 *)&beta_, accessor_c.get_pointer(),
                    (const gint_t *)&ldc);
        });
    });
}

void syr2k(cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
           float alpha, cl::sycl::buffer<float, 1> &a, int64_t lda, cl::sycl::buffer<float, 1> &b,
           int64_t ldb, float beta, cl::sycl::buffer<float, 1> &c, int64_t ldc) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char upper_lower_ = *fortran_char(upper_lower);
        const char trans_       = *fortran_char(trans);
        auto accessor_a         = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b         = b.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_c         = c.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class blis_kernel_ssyr2k>(cgh, [=]() {
            ::ssyr2k((const char *)&upper_lower_, (const char *)&trans_, (const gint_t *)&n,
                     (const gint_t *)&k, (const float *)&alpha, accessor_a.get_pointer(),
                     (const gint_t *)&lda, accessor_b.get_pointer(), (const gint_t *)&ldb,
                     (const float *)&beta, accessor_c.get_pointer(), (const gint_t *)&ldc);
        });
    });
}

void syr2k(cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
           double alpha, cl::sycl::buffer<double, 1> &a, int64_t lda,
           cl::sycl::buffer<double, 1> &b, int64_t ldb, double beta, cl::sycl::buffer<double, 1> &c,
           int64_t ldc) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char upper_lower_ = *fortran_char(upper_lower);
        const char trans_       = *fortran_char(trans);
        auto accessor_a         = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b         = b.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_c         = c.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class blis_kernel_dsyr2k>(cgh, [=]() {
            ::dsyr2k((const char *)&upper_lower_, (const char *)&trans_, (const gint_t *)&n,
                     (const gint_t *)&k, (const double *)&alpha, accessor_a.get_pointer(),
                     (const gint_t *)&lda, accessor_b.get_pointer(), (const gint_t *)&ldb,
                     (const double *)&beta, accessor_c.get_pointer(), (const gint_t *)&ldc);
        });
    });
}

void syr2k(cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
           std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
           cl::sycl::buffer<std::complex<float>, 1> &b, int64_t ldb, std::complex<float> beta,
           cl::sycl::buffer<std::complex<float>, 1> &c, int64_t ldc) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char upper_lower_ = *fortran_char(upper_lower);
        const char trans_       = *fortran_char(trans);
        float alpha_real = alpha.real(), alpha_imag = alpha.imag();
        float beta_real = beta.real(), beta_imag = beta.imag();
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b = b.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_c = c.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class blis_kernel_csyr2k>(cgh, [=]() {
            BLIS_Complex8 alpha_ = { alpha_real, alpha_imag };
            BLIS_Complex8 beta_  = { beta_real, beta_imag };
            ::csyr2k((const char *)&upper_lower_, (const char *)&trans_, (const gint_t *)&n,
                     (const gint_t *)&k, (const BLIS_Complex8 *)&alpha_, accessor_a.get_pointer(),
                     (const gint_t *)&lda, accessor_b.get_pointer(), (const gint_t *)&ldb,
                     (const BLIS_Complex8 *)&beta_, accessor_c.get_pointer(), (const gint_t *)&ldc);
        });
    });
}

void syr2k(cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
           std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
           cl::sycl::buffer<std::complex<double>, 1> &b, int64_t ldb, std::complex<double> beta,
           cl::sycl::buffer<std::complex<double>, 1> &c, int64_t ldc) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char upper_lower_ = *fortran_char(upper_lower);
        const char trans_       = *fortran_char(trans);
        double alpha_real = alpha.real(), alpha_imag = alpha.imag();
        double beta_real = beta.real(), beta_imag = beta.imag();
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b = b.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_c = c.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class blis_kernel_zsyr2k>(cgh, [=]() {
            BLIS_Complex16 alpha_ = { alpha_real, alpha_imag };
            BLIS_Complex16 beta_  = { beta_real, beta_imag };
            ::zsyr2k((const char *)&upper_lower_, (const char *)&trans_, (const gint_t *)&n,
                     (const gint_t *)&k, (const BLIS_Complex16 *)&alpha_, accessor_a.get_pointer(),
                     (const gint_t *)&lda, accessor_b.get_pointer(), (const gint_t *)&ldb,
                     (const BLIS_Complex16 *)&beta_, accessor_c.get_pointer(),
                     (const gint_t *)&ldc);
        });
    });
}

void trmm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose transa,
          diag unit_diag, int64_t m, int64_t n, float alpha, cl::sycl::buffer<float, 1> &a,
          int64_t lda, cl::sycl::buffer<float, 1> &b, int64_t ldb) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char left_right_  = *fortran_char(left_right);
        const char upper_lower_ = *fortran_char(upper_lower);
        const char transa_      = *fortran_char(transa);
        const char unit_diag_   = *fortran_char(unit_diag);
        auto accessor_a         = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b         = b.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class blis_kernel_strmm>(cgh, [=]() {
            ::strmm((const char *)&left_right_, (const char *)&upper_lower_, (const char *)&transa_,
                    (const char *)&unit_diag_, (const gint_t *)&m, (const gint_t *)&n,
                    (const float *)&alpha, accessor_a.get_pointer(), (const gint_t *)&lda,
                    accessor_b.get_pointer(), (const gint_t *)&ldb);
        });
    });
}

void trmm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose transa,
          diag unit_diag, int64_t m, int64_t n, double alpha, cl::sycl::buffer<double, 1> &a,
          int64_t lda, cl::sycl::buffer<double, 1> &b, int64_t ldb) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char left_right_  = *fortran_char(left_right);
        const char upper_lower_ = *fortran_char(upper_lower);
        const char transa_      = *fortran_char(transa);
        const char unit_diag_   = *fortran_char(unit_diag);
        auto accessor_a         = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b         = b.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class blis_kernel_dtrmm>(cgh, [=]() {
            ::dtrmm((const char *)&left_right_, (const char *)&upper_lower_, (const char *)&transa_,
                    (const char *)&unit_diag_, (const gint_t *)&m, (const gint_t *)&n,
                    (const double *)&alpha, accessor_a.get_pointer(), (const gint_t *)&lda,
                    accessor_b.get_pointer(), (const gint_t *)&ldb);
        });
    });
}

void trmm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose transa,
          diag unit_diag, int64_t m, int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &b, int64_t ldb) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char left_right_  = *fortran_char(left_right);
        const char upper_lower_ = *fortran_char(upper_lower);
        const char transa_      = *fortran_char(transa);
        const char unit_diag_   = *fortran_char(unit_diag);
        float alpha_real = alpha.real(), alpha_imag = alpha.imag();
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b = b.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class blis_kernel_ctrmm>(cgh, [=]() {
            BLIS_Complex8 alpha_ = { alpha_real, alpha_imag };
            ::ctrmm((const char *)&left_right_, (const char *)&upper_lower_, (const char *)&transa_,
                    (const char *)&unit_diag_, (const gint_t *)&m, (const gint_t *)&n,
                    (const BLIS_Complex8 *)&alpha_, accessor_a.get_pointer(), (const gint_t *)&lda,
                    accessor_b.get_pointer(), (const gint_t *)&ldb);
        });
    });
}

void trmm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose transa,
          diag unit_diag, int64_t m, int64_t n, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &b, int64_t ldb) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char left_right_  = *fortran_char(left_right);
        const char upper_lower_ = *fortran_char(upper_lower);
        const char transa_      = *fortran_char(transa);
        const char unit_diag_   = *fortran_char(unit_diag);
        double alpha_real = alpha.real(), alpha_imag = alpha.imag();
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b = b.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class blis_kernel_ztrmm>(cgh, [=]() {
            BLIS_Complex16 alpha_ = { alpha_real, alpha_imag };
            ::ztrmm((const char *)&left_right_, (const char *)&upper_lower_, (const char *)&transa_,
                    (const char *)&unit_diag_, (const gint_t *)&m, (const gint_t *)&n,
                    (const BLIS_Complex16 *)&alpha_, accessor_a.get_pointer(), (const gint_t *)&lda,
                    accessor_b.get_pointer(), (const gint_t *)&ldb);
        });
    });
}

void trsm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose transa,
          diag unit_diag, int64_t m, int64_t n, float alpha, cl::sycl::buffer<float, 1> &a,
          int64_t lda, cl::sycl::buffer<float, 1> &b, int64_t ldb) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char left_right_  = *fortran_char(left_right);
        const char upper_lower_ = *fortran_char(upper_lower);
        const char transa_      = *fortran_char(transa);
        const char unit_diag_   = *fortran_char(unit_diag);
        auto accessor_a         = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b         = b.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class blis_kernel_strsm>(cgh, [=]() {
            ::strsm((const char *)&left_right_, (const char *)&upper_lower_, (const char *)&transa_,
                    (const char *)&unit_diag_, (const gint_t *)&m, (const gint_t *)&n,
                    (const float *)&alpha, accessor_a.get_pointer(), (const gint_t *)&lda,
                    accessor_b.get_pointer(), (const gint_t *)&ldb);
        });
    });
}

void trsm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose transa,
          diag unit_diag, int64_t m, int64_t n, double alpha, cl::sycl::buffer<double, 1> &a,
          int64_t lda, cl::sycl::buffer<double, 1> &b, int64_t ldb) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char left_right_  = *fortran_char(left_right);
        const char upper_lower_ = *fortran_char(upper_lower);
        const char transa_      = *fortran_char(transa);
        const char unit_diag_   = *fortran_char(unit_diag);
        auto accessor_a         = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b         = b.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class blis_kernel_dtrsm>(cgh, [=]() {
            ::dtrsm((const char *)&left_right_, (const char *)&upper_lower_, (const char *)&transa_,
                    (const char *)&unit_diag_, (const gint_t *)&m, (const gint_t *)&n,
                    (const double *)&alpha, accessor_a.get_pointer(), (const gint_t *)&lda,
                    accessor_b.get_pointer(), (const gint_t *)&ldb);
        });
    });
}

void trsm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose transa,
          diag unit_diag, int64_t m, int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &b, int64_t ldb) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char left_right_  = *fortran_char(left_right);
        const char upper_lower_ = *fortran_char(upper_lower);
        const char transa_      = *fortran_char(transa);
        const char unit_diag_   = *fortran_char(unit_diag);
        float alpha_real = alpha.real(), alpha_imag = alpha.imag();
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b = b.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class blis_kernel_ctrsm>(cgh, [=]() {
            BLIS_Complex8 alpha_ = { alpha_real, alpha_imag };
            ::ctrsm((const char *)&left_right_, (const char *)&upper_lower_, (const char *)&transa_,
                    (const char *)&unit_diag_, (const gint_t *)&m, (const gint_t *)&n,
                    (const BLIS_Complex8 *)&alpha_, accessor_a.get_pointer(), (const gint_t *)&lda,
                    accessor_b.get_pointer(), (const gint_t *)&ldb);
        });
    });
}

void trsm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose transa,
          diag unit_diag, int64_t m, int64_t n, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &b, int64_t ldb) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char left_right_  = *fortran_char(left_right);
        const char upper_lower_ = *fortran_char(upper_lower);
        const char transa_      = *fortran_char(transa);
        const char unit_diag_   = *fortran_char(unit_diag);
        double alpha_real = alpha.real(), alpha_imag = alpha.imag();
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b = b.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class blis_kernel_ztrsm>(cgh, [=]() {
            BLIS_Complex16 alpha_ = { alpha_real, alpha_imag };
            ::ztrsm((const char *)&left_right_, (const char *)&upper_lower_, (const char *)&transa_,
                    (const char *)&unit_diag_, (const gint_t *)&m, (const gint_t *)&n,
                    (const BLIS_Complex16 *)&alpha_, accessor_a.get_pointer(), (const gint_t *)&lda,
                    accessor_b.get_pointer(), (const gint_t *)&ldb);
        });
    });
}

} // namespace blis
} // namespace onemkl
