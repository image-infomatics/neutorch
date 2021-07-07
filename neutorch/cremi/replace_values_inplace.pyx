from cython.operator cimport dereference as deref
from libcpp.map cimport map
import logging
import numpy as np
cimport cython
cimport numpy as np

# Use type template to handle different input/output datatype combinations
ctypedef fused input_type:
    np.uint8_t
    np.uint16_t
    np.uint32_t
    np.uint64_t
    np.int8_t
    np.int16_t
    np.int32_t
    np.int64_t
ctypedef fused output_type:
    np.uint8_t
    np.uint16_t
    np.uint32_t
    np.uint64_t
    np.int8_t
    np.int16_t
    np.int32_t
    np.int64_t


@cython.boundscheck(False)
@cython.wraparound(False)
def replace_values_inplace(
        np.ndarray[input_type, ndim=1, mode='c'] in_array,
        np.ndarray[input_type, ndim=1, mode='c'] old_values,
        np.ndarray[output_type, ndim=1, mode='c'] new_values,
        np.ndarray[output_type, ndim=1, mode='c'] out_array
):

    cdef Py_ssize_t i = 0
    cdef Py_ssize_t n = in_array.size
    cdef input_type * a = &in_array[0]
    cdef output_type * b = &out_array[0]

    cdef map[input_type, output_type] cmap
    for i in range(old_values.size):
        cmap[old_values[i]] = new_values[i]

    for i in range(n):
        it = cmap.find(a[i])
        if it == cmap.end():
            continue
        b[i] = deref(it).second
