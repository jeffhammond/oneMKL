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

void asum(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<float, 1> &x, int64_t incx,
          cl::sycl::buffer<float, 1> &result) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x      = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_result = result.get_access<cl::sycl::access::mode::write>(cgh);
        host_task<class blis_kernel_sasum>(cgh, [=]() {
            accessor_result[0] =
                ::sasum((const gint_t *)&n, accessor_x.get_pointer(), (const gint_t *)&incx);
        });
    });
}

void asum(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<double, 1> &x, int64_t incx,
          cl::sycl::buffer<double, 1> &result) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x      = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_result = result.get_access<cl::sycl::access::mode::write>(cgh);
        host_task<class blis_kernel_dasum>(cgh, [=]() {
            accessor_result[0] =
                ::dasum((const gint_t *)&n, accessor_x.get_pointer(), (const gint_t *)&incx);
        });
    });
}

void asum(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<std::complex<float>, 1> &x,
          int64_t incx, cl::sycl::buffer<float, 1> &result) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x      = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_result = result.get_access<cl::sycl::access::mode::write>(cgh);
        host_task<class blis_kernel_scasum>(cgh, [=]() {
            accessor_result[0] =
                ::scasum((const gint_t *)&n, accessor_x.get_pointer(), (const gint_t *)&incx);
        });
    });
}

void asum(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<std::complex<double>, 1> &x,
          int64_t incx, cl::sycl::buffer<double, 1> &result) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x      = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_result = result.get_access<cl::sycl::access::mode::write>(cgh);
        host_task<class blis_kernel_dzasum>(cgh, [=]() {
            accessor_result[0] =
                ::dzasum((const gint_t *)&n, accessor_x.get_pointer(), (const gint_t *)&incx);
        });
    });
}

void axpy(cl::sycl::queue &queue, int64_t n, float alpha, cl::sycl::buffer<float, 1> &x,
          int64_t incx, cl::sycl::buffer<float, 1> &y, int64_t incy) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class blis_kernel_saxpy>(cgh, [=]() {
            ::saxpy((const gint_t *)&n, (const float *)&alpha, accessor_x.get_pointer(),
                    (const gint_t *)&incx, accessor_y.get_pointer(), (const gint_t *)&incy);
        });
    });
}

void axpy(cl::sycl::queue &queue, int64_t n, double alpha, cl::sycl::buffer<double, 1> &x,
          int64_t incx, cl::sycl::buffer<double, 1> &y, int64_t incy) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class blis_kernel_daxpy>(cgh, [=]() {
            ::daxpy((const gint_t *)&n, (const double *)&alpha, accessor_x.get_pointer(),
                    (const gint_t *)&incx, accessor_y.get_pointer(), (const gint_t *)&incy);
        });
    });
}

void axpy(cl::sycl::queue &queue, int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx,
          cl::sycl::buffer<std::complex<float>, 1> &y, int64_t incy) {
    queue.submit([&](cl::sycl::handler &cgh) {
        float alpha_real = alpha.real(), alpha_imag = alpha.imag();
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class blis_kernel_caxpy>(cgh, [=]() {
            BLIS_Complex8 alpha_ = { alpha_real, alpha_imag };
            ::caxpy((const gint_t *)&n, (const BLIS_Complex8 *)&alpha_, accessor_x.get_pointer(),
                    (const gint_t *)&incx, accessor_y.get_pointer(), (const gint_t *)&incy);
        });
    });
}

void axpy(cl::sycl::queue &queue, int64_t n, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx,
          cl::sycl::buffer<std::complex<double>, 1> &y, int64_t incy) {
    queue.submit([&](cl::sycl::handler &cgh) {
        double alpha_real = alpha.real(), alpha_imag = alpha.imag();
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class blis_kernel_zaxpy>(cgh, [=]() {
            BLIS_Complex16 alpha_ = { alpha_real, alpha_imag };
            ::zaxpy((const gint_t *)&n, (const BLIS_Complex16 *)&alpha_, accessor_x.get_pointer(),
                    (const gint_t *)&incx, accessor_y.get_pointer(), (const gint_t *)&incy);
        });
    });
}

void copy(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<float, 1> &x, int64_t incx,
          cl::sycl::buffer<float, 1> &y, int64_t incy) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class blis_kernel_scopy>(cgh, [=]() {
            ::scopy((const gint_t *)&n, accessor_x.get_pointer(), (const gint_t *)&incx,
                    accessor_y.get_pointer(), (const gint_t *)&incy);
        });
    });
}

void copy(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<double, 1> &x, int64_t incx,
          cl::sycl::buffer<double, 1> &y, int64_t incy) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class blis_kernel_dcopy>(cgh, [=]() {
            ::dcopy((const gint_t *)&n, accessor_x.get_pointer(), (const gint_t *)&incx,
                    accessor_y.get_pointer(), (const gint_t *)&incy);
        });
    });
}

void copy(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<std::complex<float>, 1> &x,
          int64_t incx, cl::sycl::buffer<std::complex<float>, 1> &y, int64_t incy) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class blis_kernel_ccopy>(cgh, [=]() {
            ::ccopy((const gint_t *)&n, accessor_x.get_pointer(), (const gint_t *)&incx,
                    accessor_y.get_pointer(), (const gint_t *)&incy);
        });
    });
}

void copy(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<std::complex<double>, 1> &x,
          int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y, int64_t incy) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class blis_kernel_zcopy>(cgh, [=]() {
            ::zcopy((const gint_t *)&n, accessor_x.get_pointer(), (const gint_t *)&incx,
                    accessor_y.get_pointer(), (const gint_t *)&incy);
        });
    });
}

void dot(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<float, 1> &x, int64_t incx,
         cl::sycl::buffer<float, 1> &y, int64_t incy, cl::sycl::buffer<float, 1> &result) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x      = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y      = y.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_result = result.get_access<cl::sycl::access::mode::write>(cgh);
        host_task<class blis_kernel_sdot>(cgh, [=]() {
            accessor_result[0] =
                ::sdot((const gint_t *)&n, accessor_x.get_pointer(), (const gint_t *)&incx,
                       accessor_y.get_pointer(), (const gint_t *)&incy);
        });
    });
}

void dot(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<double, 1> &x, int64_t incx,
         cl::sycl::buffer<double, 1> &y, int64_t incy, cl::sycl::buffer<double, 1> &result) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x      = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y      = y.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_result = result.get_access<cl::sycl::access::mode::write>(cgh);
        host_task<class blis_kernel_ddot>(cgh, [=]() {
            accessor_result[0] =
                ::ddot((const gint_t *)&n, accessor_x.get_pointer(), (const gint_t *)&incx,
                       accessor_y.get_pointer(), (const gint_t *)&incy);
        });
    });
}

void dot(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<float, 1> &x, int64_t incx,
         cl::sycl::buffer<float, 1> &y, int64_t incy, cl::sycl::buffer<double, 1> &result) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x      = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y      = y.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_result = result.get_access<cl::sycl::access::mode::write>(cgh);
        host_task<class blis_kernel_dsdot>(cgh, [=]() {
            accessor_result[0] =
                ::dsdot((const gint_t *)&n, accessor_x.get_pointer(), (const gint_t *)&incx,
                        accessor_y.get_pointer(), (const gint_t *)&incy);
        });
    });
}

void dotc(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<std::complex<float>, 1> &x,
          int64_t incx, cl::sycl::buffer<std::complex<float>, 1> &y, int64_t incy,
          cl::sycl::buffer<std::complex<float>, 1> &result) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x      = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y      = y.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_result = result.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class blis_kernel_cdotc>(cgh, [=]() {
            ::cdotc(accessor_result.get_pointer(), (const gint_t *)&n, accessor_x.get_pointer(),
                    (const gint_t *)&incx, accessor_y.get_pointer(), (const gint_t *)&incy);
        });
    });
}

void dotc(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<std::complex<double>, 1> &x,
          int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y, int64_t incy,
          cl::sycl::buffer<std::complex<double>, 1> &result) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x      = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y      = y.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_result = result.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class blis_kernel_zdotc>(cgh, [=]() {
            ::zdotc(accessor_result.get_pointer(), (const gint_t *)&n, accessor_x.get_pointer(),
                    (const gint_t *)&incx, accessor_y.get_pointer(), (const gint_t *)&incy);
        });
    });
}

void dotu(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<std::complex<float>, 1> &x,
          int64_t incx, cl::sycl::buffer<std::complex<float>, 1> &y, int64_t incy,
          cl::sycl::buffer<std::complex<float>, 1> &result) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x      = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y      = y.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_result = result.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class blis_kernel_cdotu>(cgh, [=]() {
            ::cdotu(accessor_result.get_pointer(), (const gint_t *)&n, accessor_x.get_pointer(),
                    (const gint_t *)&incx, accessor_y.get_pointer(), (const gint_t *)&incy);
        });
    });
}

void dotu(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<std::complex<double>, 1> &x,
          int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y, int64_t incy,
          cl::sycl::buffer<std::complex<double>, 1> &result) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x      = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y      = y.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_result = result.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class blis_kernel_zdotu>(cgh, [=]() {
            ::zdotu(accessor_result.get_pointer(), (const gint_t *)&n, accessor_x.get_pointer(),
                    (const gint_t *)&incx, accessor_y.get_pointer(), (const gint_t *)&incy);
        });
    });
}

void iamin(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<float, 1> &x, int64_t incx,
           cl::sycl::buffer<int64_t, 1> &result) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x      = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_result = result.get_access<cl::sycl::access::mode::write>(cgh);
        host_task<class blis_kernel_isamin>(cgh, [=]() {
            accessor_result[0] =
                ::cblas_isamin((gint_t)n, accessor_x.get_pointer(), (gint_t)incx);
        });
    });
}

void iamin(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<double, 1> &x, int64_t incx,
           cl::sycl::buffer<int64_t, 1> &result) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x      = x.template get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_result = result.template get_access<cl::sycl::access::mode::write>(cgh);
        host_task<class blis_kernel_idamin>(cgh, [=]() {
            accessor_result[0] =
                ::cblas_idamin((const gint_t)n, accessor_x.get_pointer(), (const gint_t)incx);
        });
    });
}

void iamin(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<std::complex<float>, 1> &x,
           int64_t incx, cl::sycl::buffer<int64_t, 1> &result) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x      = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_result = result.get_access<cl::sycl::access::mode::write>(cgh);
        host_task<class blis_kernel_icamin>(cgh, [=]() {
            accessor_result[0] =
                ::cblas_icamin((gint_t)n, accessor_x.get_pointer(), (gint_t)incx);
        });
    });
}

void iamin(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<std::complex<double>, 1> &x,
           int64_t incx, cl::sycl::buffer<int64_t, 1> &result) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x      = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_result = result.get_access<cl::sycl::access::mode::write>(cgh);
        host_task<class blis_kernel_izamin>(cgh, [=]() {
            accessor_result[0] =
                ::cblas_izamin((gint_t)n, accessor_x.get_pointer(), (gint_t)incx);
        });
    });
}

void iamax(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<float, 1> &x, int64_t incx,
           cl::sycl::buffer<int64_t, 1> &result) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x      = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_result = result.get_access<cl::sycl::access::mode::write>(cgh);
        host_task<class blis_kernel_isamax>(cgh, [=]() {
            accessor_result[0] =
                ::cblas_isamax((gint_t)n, accessor_x.get_pointer(), (gint_t)incx);
        });
    });
}

void iamax(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<double, 1> &x, int64_t incx,
           cl::sycl::buffer<int64_t, 1> &result) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x      = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_result = result.get_access<cl::sycl::access::mode::write>(cgh);
        host_task<class blis_kernel_idamax>(cgh, [=]() {
            accessor_result[0] =
                ::cblas_idamax((gint_t)n, accessor_x.get_pointer(), (gint_t)incx);
        });
    });
}

void iamax(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<std::complex<float>, 1> &x,
           int64_t incx, cl::sycl::buffer<int64_t, 1> &result) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x      = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_result = result.get_access<cl::sycl::access::mode::write>(cgh);
        host_task<class blis_kernel_icamax>(cgh, [=]() {
            accessor_result[0] =
                ::cblas_icamax((gint_t)n, accessor_x.get_pointer(), (gint_t)incx);
        });
    });
}

void iamax(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<std::complex<double>, 1> &x,
           int64_t incx, cl::sycl::buffer<int64_t, 1> &result) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x      = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_result = result.get_access<cl::sycl::access::mode::write>(cgh);
        host_task<class blis_kernel_izamax>(cgh, [=]() {
            accessor_result[0] =
                ::cblas_izamax((gint_t)n, accessor_x.get_pointer(), (gint_t)incx);
        });
    });
}

void nrm2(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<float, 1> &x, int64_t incx,
          cl::sycl::buffer<float, 1> &result) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x      = x.template get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_result = result.template get_access<cl::sycl::access::mode::write>(cgh);
        host_task<class blis_kernel_snrm2>(cgh, [=]() {
            accessor_result[0] =
                ::snrm2((const gint_t *)&n, accessor_x.get_pointer(), (const gint_t *)&incx);
        });
    });
}

void nrm2(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<double, 1> &x, int64_t incx,
          cl::sycl::buffer<double, 1> &result) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x      = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_result = result.get_access<cl::sycl::access::mode::write>(cgh);
        host_task<class blis_kernel_dnrm2>(cgh, [=]() {
            accessor_result[0] =
                ::dnrm2((const gint_t *)&n, accessor_x.get_pointer(), (const gint_t *)&incx);
        });
    });
}

void nrm2(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<std::complex<float>, 1> &x,
          int64_t incx, cl::sycl::buffer<float, 1> &result) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x      = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_result = result.get_access<cl::sycl::access::mode::write>(cgh);
        host_task<class blis_kernel_scnrm2>(cgh, [=]() {
            accessor_result[0] =
                ::scnrm2((const gint_t *)&n, accessor_x.get_pointer(), (const gint_t *)&incx);
        });
    });
}

void nrm2(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<std::complex<double>, 1> &x,
          int64_t incx, cl::sycl::buffer<double, 1> &result) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x      = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_result = result.get_access<cl::sycl::access::mode::write>(cgh);
        host_task<class blis_kernel_dznrm2>(cgh, [=]() {
            accessor_result[0] =
                ::dznrm2((const gint_t *)&n, accessor_x.get_pointer(), (const gint_t *)&incx);
        });
    });
}

void rot(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<float, 1> &x, int64_t incx,
         cl::sycl::buffer<float, 1> &y, int64_t incy, float c, float s) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x = x.get_access<cl::sycl::access::mode::read_write>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class blis_kernel_srot>(cgh, [=]() {
            ::srot((const gint_t *)&n, accessor_x.get_pointer(), (const gint_t *)&incx,
                   accessor_y.get_pointer(), (const gint_t *)&incy, &c, &s);
        });
    });
}

void rot(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<double, 1> &x, int64_t incx,
         cl::sycl::buffer<double, 1> &y, int64_t incy, double c, double s) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x = x.get_access<cl::sycl::access::mode::read_write>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class blis_kernel_drot>(cgh, [=]() {
            ::drot((const gint_t *)&n, accessor_x.get_pointer(), (const gint_t *)&incx,
                   accessor_y.get_pointer(), (const gint_t *)&incy, &c, &s);
        });
    });
}

void rot(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<std::complex<float>, 1> &x,
         int64_t incx, cl::sycl::buffer<std::complex<float>, 1> &y, int64_t incy, float c,
         float s) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x = x.get_access<cl::sycl::access::mode::read_write>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class blis_kernel_csrot>(cgh, [=]() {
            ::csrot((const gint_t *)&n, accessor_x.get_pointer(), (const gint_t *)&incx,
                    accessor_y.get_pointer(), (const gint_t *)&incy, &c, &s);
        });
    });
}

void rot(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<std::complex<double>, 1> &x,
         int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y, int64_t incy, double c,
         double s) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x = x.get_access<cl::sycl::access::mode::read_write>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class blis_kernel_zdrot>(cgh, [=]() {
            ::zdrot((const gint_t *)&n, accessor_x.get_pointer(), (const gint_t *)&incx,
                    accessor_y.get_pointer(), (const gint_t *)&incy, &c, &s);
        });
    });
}

void rotg(cl::sycl::queue &queue, cl::sycl::buffer<float, 1> &a, cl::sycl::buffer<float, 1> &b,
          cl::sycl::buffer<float, 1> &c, cl::sycl::buffer<float, 1> &s) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_a = a.get_access<cl::sycl::access::mode::read_write>(cgh);
        auto accessor_b = b.get_access<cl::sycl::access::mode::read_write>(cgh);
        auto accessor_c = c.get_access<cl::sycl::access::mode::read_write>(cgh);
        auto accessor_s = s.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class blis_kernel_srotg>(cgh, [=]() {
            ::srotg(accessor_a.get_pointer(), accessor_b.get_pointer(), accessor_c.get_pointer(),
                    accessor_s.get_pointer());
        });
    });
}

void rotg(cl::sycl::queue &queue, cl::sycl::buffer<double, 1> &a, cl::sycl::buffer<double, 1> &b,
          cl::sycl::buffer<double, 1> &c, cl::sycl::buffer<double, 1> &s) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_a = a.get_access<cl::sycl::access::mode::read_write>(cgh);
        auto accessor_b = b.get_access<cl::sycl::access::mode::read_write>(cgh);
        auto accessor_c = c.get_access<cl::sycl::access::mode::read_write>(cgh);
        auto accessor_s = s.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class blis_kernel_drotg>(cgh, [=]() {
            ::drotg(accessor_a.get_pointer(), accessor_b.get_pointer(), accessor_c.get_pointer(),
                    accessor_s.get_pointer());
        });
    });
}

void rotg(cl::sycl::queue &queue, cl::sycl::buffer<std::complex<float>, 1> &a,
          cl::sycl::buffer<std::complex<float>, 1> &b, cl::sycl::buffer<float, 1> &c,
          cl::sycl::buffer<std::complex<float>, 1> &s) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_a = a.get_access<cl::sycl::access::mode::read_write>(cgh);
        auto accessor_b = b.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_c = c.get_access<cl::sycl::access::mode::read_write>(cgh);
        auto accessor_s = s.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class blis_kernel_crotg>(cgh, [=]() {
            ::crotg(accessor_a.get_pointer(), accessor_b.get_pointer(), accessor_c.get_pointer(),
                    accessor_s.get_pointer());
        });
    });
}

void rotg(cl::sycl::queue &queue, cl::sycl::buffer<std::complex<double>, 1> &a,
          cl::sycl::buffer<std::complex<double>, 1> &b, cl::sycl::buffer<double, 1> &c,
          cl::sycl::buffer<std::complex<double>, 1> &s) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_a = a.get_access<cl::sycl::access::mode::read_write>(cgh);
        auto accessor_b = b.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_c = c.get_access<cl::sycl::access::mode::read_write>(cgh);
        auto accessor_s = s.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class blis_kernel_zrotg>(cgh, [=]() {
            ::zrotg(accessor_a.get_pointer(), accessor_b.get_pointer(), accessor_c.get_pointer(),
                    accessor_s.get_pointer());
        });
    });
}

void rotm(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<float, 1> &x, int64_t incx,
          cl::sycl::buffer<float, 1> &y, int64_t incy, cl::sycl::buffer<float, 1> &param) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x     = x.get_access<cl::sycl::access::mode::read_write>(cgh);
        auto accessor_y     = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        auto accessor_param = param.get_access<cl::sycl::access::mode::read>(cgh);
        host_task<class blis_kernel_srotm>(cgh, [=]() {
            ::srotm((const gint_t *)&n, accessor_x.get_pointer(), (const gint_t *)&incx,
                    accessor_y.get_pointer(), (const gint_t *)&incy, accessor_param.get_pointer());
        });
    });
}

void rotm(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<double, 1> &x, int64_t incx,
          cl::sycl::buffer<double, 1> &y, int64_t incy, cl::sycl::buffer<double, 1> &param) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x     = x.get_access<cl::sycl::access::mode::read_write>(cgh);
        auto accessor_y     = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        auto accessor_param = param.get_access<cl::sycl::access::mode::read>(cgh);
        host_task<class blis_kernel_drotm>(cgh, [=]() {
            ::drotm((const gint_t *)&n, accessor_x.get_pointer(), (const gint_t *)&incx,
                    accessor_y.get_pointer(), (const gint_t *)&incy, accessor_param.get_pointer());
        });
    });
}

void rotmg(cl::sycl::queue &queue, cl::sycl::buffer<float, 1> &d1, cl::sycl::buffer<float, 1> &d2,
           cl::sycl::buffer<float, 1> &x1, float y1, cl::sycl::buffer<float, 1> &param) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_d1    = d1.get_access<cl::sycl::access::mode::read_write>(cgh);
        auto accessor_d2    = d2.get_access<cl::sycl::access::mode::read_write>(cgh);
        auto accessor_x1    = x1.get_access<cl::sycl::access::mode::read_write>(cgh);
        auto accessor_param = param.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class blis_kernel_srotmg>(cgh, [=]() {
            ::srotmg(accessor_d1.get_pointer(), accessor_d2.get_pointer(),
                     accessor_x1.get_pointer(), (float *)&y1, accessor_param.get_pointer());
        });
    });
}

void rotmg(cl::sycl::queue &queue, cl::sycl::buffer<double, 1> &d1, cl::sycl::buffer<double, 1> &d2,
           cl::sycl::buffer<double, 1> &x1, double y1, cl::sycl::buffer<double, 1> &param) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_d1    = d1.get_access<cl::sycl::access::mode::read_write>(cgh);
        auto accessor_d2    = d2.get_access<cl::sycl::access::mode::read_write>(cgh);
        auto accessor_x1    = x1.get_access<cl::sycl::access::mode::read_write>(cgh);
        auto accessor_param = param.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class blis_kernel_drotmg>(cgh, [=]() {
            ::drotmg(accessor_d1.get_pointer(), accessor_d2.get_pointer(),
                     accessor_x1.get_pointer(), (double *)&y1, accessor_param.get_pointer());
        });
    });
}

void scal(cl::sycl::queue &queue, int64_t n, float alpha, cl::sycl::buffer<float, 1> &x,
          int64_t incx) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x = x.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class blis_kernel_sscal>(cgh, [=]() {
            ::sscal((const gint_t *)&n, (const float *)&alpha, accessor_x.get_pointer(),
                    (const gint_t *)&incx);
        });
    });
}

void scal(cl::sycl::queue &queue, int64_t n, double alpha, cl::sycl::buffer<double, 1> &x,
          int64_t incx) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x = x.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class blis_kernel_dscal>(cgh, [=]() {
            ::dscal((const gint_t *)&n, (const double *)&alpha, accessor_x.get_pointer(),
                    (const gint_t *)&incx);
        });
    });
}

void scal(cl::sycl::queue &queue, int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx) {
    queue.submit([&](cl::sycl::handler &cgh) {
        float alpha_real = alpha.real(), alpha_imag = alpha.imag();
        auto accessor_x = x.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class blis_kernel_cscal>(cgh, [=]() {
            BLIS_Complex8 alpha_ = { alpha_real, alpha_imag };
            ::cscal((const gint_t *)&n, (const BLIS_Complex8 *)&alpha_, accessor_x.get_pointer(),
                    (const gint_t *)&incx);
        });
    });
}

void scal(cl::sycl::queue &queue, int64_t n, float alpha,
          cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x = x.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class blis_kernel_csscal>(cgh, [=]() {
            ::csscal((const gint_t *)&n, (const float *)&alpha, accessor_x.get_pointer(),
                     (const gint_t *)&incx);
        });
    });
}

void scal(cl::sycl::queue &queue, int64_t n, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx) {
    queue.submit([&](cl::sycl::handler &cgh) {
        double alpha_real = alpha.real(), alpha_imag = alpha.imag();
        auto accessor_x = x.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class blis_kernel_zscal>(cgh, [=]() {
            BLIS_Complex16 alpha_ = { alpha_real, alpha_imag };
            ::zscal((const gint_t *)&n, (const BLIS_Complex16 *)&alpha_, accessor_x.get_pointer(),
                    (const gint_t *)&incx);
        });
    });
}

void scal(cl::sycl::queue &queue, int64_t n, double alpha,
          cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x = x.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class blis_kernel_zdscal>(cgh, [=]() {
            ::zdscal((const gint_t *)&n, (const double *)&alpha, accessor_x.get_pointer(),
                     (const gint_t *)&incx);
        });
    });
}

void sdsdot(cl::sycl::queue &queue, int64_t n, float sb, cl::sycl::buffer<float, 1> &x,
            int64_t incx, cl::sycl::buffer<float, 1> &y, int64_t incy,
            cl::sycl::buffer<float, 1> &result) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x      = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y      = y.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_result = result.get_access<cl::sycl::access::mode::write>(cgh);
        host_task<class blis_kernel_sdsdot>(cgh, [=]() {
            accessor_result[0] =
                ::sdsdot((const gint_t *)&n, (const float *)&sb, accessor_x.get_pointer(),
                         (const gint_t *)&incx, accessor_y.get_pointer(), (const gint_t *)&incy);
        });
    });
}

void swap(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<float, 1> &x, int64_t incx,
          cl::sycl::buffer<float, 1> &y, int64_t incy) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x = x.get_access<cl::sycl::access::mode::read_write>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class blis_kernel_sswap>(cgh, [=]() {
            ::sswap((const gint_t *)&n, accessor_x.get_pointer(), (const gint_t *)&incx,
                    accessor_y.get_pointer(), (const gint_t *)&incy);
        });
    });
}

void swap(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<double, 1> &x, int64_t incx,
          cl::sycl::buffer<double, 1> &y, int64_t incy) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x = x.get_access<cl::sycl::access::mode::read_write>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class blis_kernel_dswap>(cgh, [=]() {
            ::dswap((const gint_t *)&n, accessor_x.get_pointer(), (const gint_t *)&incx,
                    accessor_y.get_pointer(), (const gint_t *)&incy);
        });
    });
}

void swap(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<std::complex<float>, 1> &x,
          int64_t incx, cl::sycl::buffer<std::complex<float>, 1> &y, int64_t incy) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x = x.get_access<cl::sycl::access::mode::read_write>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class blis_kernel_cswap>(cgh, [=]() {
            ::cswap((const gint_t *)&n, accessor_x.get_pointer(), (const gint_t *)&incx,
                    accessor_y.get_pointer(), (const gint_t *)&incy);
        });
    });
}

void swap(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<std::complex<double>, 1> &x,
          int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y, int64_t incy) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x = x.get_access<cl::sycl::access::mode::read_write>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class blis_kernel_zswap>(cgh, [=]() {
            ::zswap((const gint_t *)&n, accessor_x.get_pointer(), (const gint_t *)&incx,
                    accessor_y.get_pointer(), (const gint_t *)&incy);
        });
    });
}

} // namespace blis
} // namespace onemkl
