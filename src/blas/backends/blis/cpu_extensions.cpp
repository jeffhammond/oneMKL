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
#include "fp16.hpp"

namespace onemkl {
namespace blis {

template <typename T_src, typename T_dest>
static inline void copy_mat(T_src &src, transpose trans, int64_t row, int64_t col, int64_t ld,
                            T_dest off, T_dest *&dest) {
    int64_t i, j;
    if (trans == transpose::N) {
        for (j = 0; j < col; j++) {
            for (i = 0; i < row; i++) {
                dest[i + ld * j] = (T_dest)src[i + ld * j] - off;
            }
        }
    }
    else {
        for (i = 0; i < row; i++) {
            for (j = 0; j < col; j++) {
                dest[i * ld + j] = (T_dest)src[i * ld + j] - off;
            }
        }
    }
}

template <typename T_src, typename T_dest, typename T_off>
static inline void copy_mat(T_src &src, int64_t row, int64_t col, int64_t ld, offset off_kind,
                            T_off off, T_dest &dest) {
    using T_data = typename std::remove_reference<decltype(dest[0])>::type;
    int64_t i, j;
    T_data tmp;

    if (off_kind == offset::F) {
        tmp = off[0];
        for (j = 0; j < col; j++) {
            for (i = 0; i < row; i++) {
                dest[i + ld * j] = tmp + (T_data)src[i + ld * j];
            }
        }
    }
    else if (off_kind == offset::C) {
        for (j = 0; j < col; j++) {
            for (i = 0; i < row; i++) {
                tmp              = off[i];
                dest[i + ld * j] = tmp + (T_data)src[i + ld * j];
            }
        }
    }
    else {
        for (j = 0; j < col; j++) {
            tmp = off[j];
            for (i = 0; i < row; i++) {
                dest[i + ld * j] = tmp + (T_data)src[i + ld * j];
            }
        }
    }
}

void gemm(cl::sycl::queue &queue, transpose transa, transpose transb, int64_t m, int64_t n,
          int64_t k, half alpha, cl::sycl::buffer<half, 1> &a, int64_t lda,
          cl::sycl::buffer<half, 1> &b, int64_t ldb, half beta, cl::sycl::buffer<half, 1> &c,
          int64_t ldc) {
    auto a_fp16 = a.reinterpret<fp16, 1>(a.get_range());
    auto b_fp16 = b.reinterpret<fp16, 1>(b.get_range());
    auto c_fp16 = c.reinterpret<fp16, 1>(c.get_range());

    queue.submit([&](cl::sycl::handler &cgh) {
        const char transa_ = *fortran_char(transa);
        const char transb_ = *fortran_char(transb);
        float f32_alpha    = (float)alpha;
        float f32_beta     = (float)beta;
        auto accessor_a    = a_fp16.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b    = b_fp16.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_c    = c_fp16.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class blis_kernel_hgemm>(cgh, [=]() {
            int64_t sizea, sizeb, sizec;
            sizea = (transa == transpose::N) ? lda * k : lda * m;
            sizeb = (transb == transpose::N) ? ldb * n : ldb * k;
            sizec = ldc * n;
            // copy A, B and C to float
            float *f32_a = (float *)::malloc(sizeof(float) * sizea);
            float *f32_b = (float *)::malloc(sizeof(float) * sizeb);
            float *f32_c = (float *)::malloc(sizeof(float) * sizec);
            copy_mat(accessor_a, transa, m, k, lda, 0.0f, f32_a);
            copy_mat(accessor_b, transb, k, n, ldb, 0.0f, f32_b);
            copy_mat(accessor_c, transpose::N, m, n, ldc, 0.0f, f32_c);
            ::sgemm((const char *)&transa_, (const char *)&transb_, (const gint_t *)&m,
                    (const gint_t *)&n, (const gint_t *)&k, (const float *)&f32_alpha, f32_a,
                    (const gint_t *)&lda, f32_b, (const gint_t *)&ldb, (const float *)&f32_beta,
                    f32_c, (const gint_t *)&ldc);
            // copy C back to half
            fp16 co = 0.0f;
            copy_mat(f32_c, m, n, ldc, offset::F, &co, accessor_c);
            ::free(f32_a);
            ::free(f32_b);
            ::free(f32_c);
        });
    });
}

void gemm_ext(cl::sycl::queue &queue, transpose transa, transpose transb, int64_t m, int64_t n,
              int64_t k, float alpha, cl::sycl::buffer<float, 1> &a, int64_t lda,
              cl::sycl::buffer<float, 1> &b, int64_t ldb, float beta, cl::sycl::buffer<float, 1> &c,
              int64_t ldc) {
    gemm(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void gemm_ext(cl::sycl::queue &queue, transpose transa, transpose transb, int64_t m, int64_t n,
              int64_t k, double alpha, cl::sycl::buffer<double, 1> &a, int64_t lda,
              cl::sycl::buffer<double, 1> &b, int64_t ldb, double beta,
              cl::sycl::buffer<double, 1> &c, int64_t ldc) {
    gemm(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void gemm_ext(cl::sycl::queue &queue, transpose transa, transpose transb, int64_t m, int64_t n,
              int64_t k, std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
              int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &b, int64_t ldb,
              std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &c, int64_t ldc) {
    gemm(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void gemm_ext(cl::sycl::queue &queue, transpose transa, transpose transb, int64_t m, int64_t n,
              int64_t k, std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
              int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &b, int64_t ldb,
              std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c,
              int64_t ldc) {
    gemm(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void gemm_ext(cl::sycl::queue &queue, transpose transa, transpose transb, int64_t m, int64_t n,
              int64_t k, half alpha, cl::sycl::buffer<half, 1> &a, int64_t lda,
              cl::sycl::buffer<half, 1> &b, int64_t ldb, half beta, cl::sycl::buffer<half, 1> &c,
              int64_t ldc) {
    gemm(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void gemm_ext(cl::sycl::queue &queue, transpose transa, transpose transb, int64_t m, int64_t n,
              int64_t k, float alpha, cl::sycl::buffer<half, 1> &a, int64_t lda,
              cl::sycl::buffer<half, 1> &b, int64_t ldb, float beta, cl::sycl::buffer<float, 1> &c,
              int64_t ldc) {
    auto a_fp16 = a.reinterpret<fp16, 1>(a.get_range());
    auto b_fp16 = b.reinterpret<fp16, 1>(b.get_range());
    queue.submit([&](cl::sycl::handler &cgh) {
        const char transa_ = *fortran_char(transa);
        const char transb_ = *fortran_char(transb);
        auto accessor_a    = a_fp16.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b    = b_fp16.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_c    = c.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class blis_kernel_gemm_f16f16f32>(cgh, [=]() {
            int64_t sizea, sizeb;
            sizea = (transa == transpose::N) ? lda * k : lda * m;
            sizeb = (transb == transpose::N) ? ldb * n : ldb * k;
            // copy A and B to float
            float *f32_a = (float *)::malloc(sizeof(float) * sizea);
            float *f32_b = (float *)::malloc(sizeof(float) * sizeb);
            copy_mat(accessor_a, transa, m, k, lda, 0.0f, f32_a);
            copy_mat(accessor_b, transb, k, n, ldb, 0.0f, f32_b);
            ::sgemm((const char *)&transa_, (const char *)&transb_, (const gint_t *)&m,
                    (const gint_t *)&n, (const gint_t *)&k, (const float *)&alpha, f32_a,
                    (const gint_t *)&lda, f32_b, (const gint_t *)&ldb, (const float *)&beta,
                    accessor_c.get_pointer(), (const gint_t *)&ldc);
            ::free(f32_a);
            ::free(f32_b);
        });
    });
}

void gemm_ext(cl::sycl::queue &queue, transpose transa, transpose transb, offset offsetc, int64_t m,
              int64_t n, int64_t k, float alpha, cl::sycl::buffer<int8_t, 1> &a, int64_t lda,
              int8_t ao, cl::sycl::buffer<uint8_t, 1> &b, int64_t ldb, uint8_t bo, float beta,
              cl::sycl::buffer<int32_t, 1> &c, int64_t ldc, cl::sycl::buffer<int32_t, 1> &co) {
    throw std::runtime_error("Not implemented for BLIS");
}

void gemmt(cl::sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb, int64_t n,
           int64_t k, float alpha, cl::sycl::buffer<float, 1> &a, int64_t lda,
           cl::sycl::buffer<float, 1> &b, int64_t ldb, float beta, cl::sycl::buffer<float, 1> &c,
           int64_t ldc) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char upper_lower_ = *fortran_char(upper_lower);
        const char transa_      = *fortran_char(transa);
        const char transb_      = *fortran_char(transb);
        auto accessor_a         = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b         = b.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_c         = c.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class blis_kernel_sgemmt>(cgh, [=]() {
            ::sgemmt((const char *)&upper_lower_, (const char *)&transa_, (const char *)&transb_,
                     (const gint_t *)&n, (const gint_t *)&k, (const float *)&alpha,
                     accessor_a.get_pointer(), (const gint_t *)&lda, accessor_b.get_pointer(),
                     (const gint_t *)&ldb, (const float *)&beta, accessor_c.get_pointer(),
                     (const gint_t *)&ldc);
        });
    });
}

void gemmt(cl::sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb, int64_t n,
           int64_t k, double alpha, cl::sycl::buffer<double, 1> &a, int64_t lda,
           cl::sycl::buffer<double, 1> &b, int64_t ldb, double beta, cl::sycl::buffer<double, 1> &c,
           int64_t ldc) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char upper_lower_ = *fortran_char(upper_lower);
        const char transa_      = *fortran_char(transa);
        const char transb_      = *fortran_char(transb);
        auto accessor_a         = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b         = b.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_c         = c.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class blis_kernel_dgemmt>(cgh, [=]() {
            ::dgemmt((const char *)&upper_lower_, (const char *)&transa_, (const char *)&transb_,
                     (const gint_t *)&n, (const gint_t *)&k, (const double *)&alpha,
                     accessor_a.get_pointer(), (const gint_t *)&lda, accessor_b.get_pointer(),
                     (const gint_t *)&ldb, (const double *)&beta, accessor_c.get_pointer(),
                     (const gint_t *)&ldc);
        });
    });
}

void gemmt(cl::sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb, int64_t n,
           int64_t k, std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
           int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &b, int64_t ldb,
           std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &c, int64_t ldc) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char upper_lower_ = *fortran_char(upper_lower);
        const char transa_      = *fortran_char(transa);
        const char transb_      = *fortran_char(transb);
        float alpha_real = alpha.real(), alpha_imag = alpha.imag();
        float beta_real = beta.real(), beta_imag = beta.imag();
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b = b.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_c = c.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class blis_kernel_cgemmt>(cgh, [=]() {
            BLIS_Complex8 alpha_ = { alpha_real, alpha_imag };
            BLIS_Complex8 beta_  = { beta_real, beta_imag };
            ::cgemmt((const char *)&upper_lower_, (const char *)&transa_, (const char *)&transb_,
                     (const gint_t *)&n, (const gint_t *)&k, (const BLIS_Complex8 *)&alpha_,
                     accessor_a.get_pointer(), (const gint_t *)&lda, accessor_b.get_pointer(),
                     (const gint_t *)&ldb, (const BLIS_Complex8 *)&beta_, accessor_c.get_pointer(),
                     (const gint_t *)&ldc);
        });
    });
}

void gemmt(cl::sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb, int64_t n,
           int64_t k, std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
           int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &b, int64_t ldb,
           std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c, int64_t ldc) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char upper_lower_ = *fortran_char(upper_lower);
        const char transa_      = *fortran_char(transa);
        const char transb_      = *fortran_char(transb);
        double alpha_real = alpha.real(), alpha_imag = alpha.imag();
        double beta_real = beta.real(), beta_imag = beta.imag();
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b = b.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_c = c.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class blis_kernel_zgemmt>(cgh, [=]() {
            BLIS_Complex16 alpha_ = { alpha_real, alpha_imag };
            BLIS_Complex16 beta_  = { beta_real, beta_imag };
            ::zgemmt((const char *)&upper_lower_, (const char *)&transa_, (const char *)&transb_,
                     (const gint_t *)&n, (const gint_t *)&k, (const BLIS_Complex16 *)&alpha_,
                     accessor_a.get_pointer(), (const gint_t *)&lda, accessor_b.get_pointer(),
                     (const gint_t *)&ldb, (const BLIS_Complex16 *)&beta_, accessor_c.get_pointer(),
                     (const gint_t *)&ldc);
        });
    });
}

} // namespace blis
} // namespace onemkl
