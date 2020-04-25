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

#ifndef _BLIS_COMMON_HPP_
#define _BLIS_COMMON_HPP_

#define BLIS_Complex8  struct { float real; float imag; }
#define BLIS_Complex16 struct { double real; double imag; }

#include <CL/sycl.hpp>
#include <complex>

#include "blis.h"

#include "onemkl/blas/detail/blis/onemkl_blas_blis.hpp"
#include "onemkl/types.hpp"

namespace onemkl {
namespace blis {

// host_task automatically uses run_on_host_intel if it is supported by the
//  compiler. Otherwise, it falls back to single_task.
template <typename K, typename H, typename F>
static inline auto host_task_internal(H &cgh, F f, int) -> decltype(cgh.run_on_host_intel(f)) {
    return cgh.run_on_host_intel(f);
}

template <typename K, typename H, typename F>
static inline void host_task_internal(H &cgh, F f, long) {
    cgh.template single_task<K>(f);
}

template <typename K, typename H, typename F>
static inline void host_task(H &cgh, F f) {
    (void)host_task_internal<K>(cgh, f, 0);
}

// Conversion functions to traditional Fortran characters.
inline const char *fortran_char(transpose t) {
    if (t == transpose::nontrans)
        return "N";
    if (t == transpose::trans)
        return "T";
    if (t == transpose::conjtrans)
        return "C";
    return "N";
}

inline const char *fortran_char(offset t) {
    if (t == offset::fix)
        return "F";
    if (t == offset::row)
        return "R";
    if (t == offset::column)
        return "C";
    return "N";
}

inline const char *fortran_char(uplo u) {
    if (u == uplo::upper)
        return "U";
    if (u == uplo::lower)
        return "L";
    return "U";
}

inline const char *fortran_char(diag d) {
    if (d == diag::nonunit)
        return "N";
    if (d == diag::unit)
        return "U";
    return "N";
}

inline const char *fortran_char(side s) {
    if (s == side::left)
        return "L";
    if (s == side::right)
        return "R";
    return "L";
}

} // namespace blis
} // namespace onemkl

#endif //_BLIS_COMMON_HPP_
