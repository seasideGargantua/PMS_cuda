#include <torch/extension.h>
#include "bindings.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("spherical_harmonics_3d_forward", &compute_3dsh_forward_tensor);
  m.def("spherical_harmonics_3d_backward", &compute_3dsh_backward_tensor);
}