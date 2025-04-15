#include <torch/extension.h>
#include "bindings.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("initiate_cost", &pms::initiate_cost_tensor);
  m.def("propagation", &pms::propagation_tensor);
}