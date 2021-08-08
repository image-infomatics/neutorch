#include <pybind11/pybind11.h>


PYBIND11_MODULE(libneutorch, m) {
    m.doc() = R"pbdoc(
        libneutorch
        -----------------------
        .. currentmodule:: libneutorch
        .. autosummary::
           :toctree: _generate
           warp3d
    )pbdoc";

    m.def("warp3d", &warp3d, R"pbdoc(
        Warp 3d image 

        used for patch augmentation.
    )pbdoc");

}