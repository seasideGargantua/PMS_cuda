#include <torch/extension.h>
#include "bindings.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("propagation", &propagation_tensor);
}