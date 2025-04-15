from typing import Callable


def _make_lazy_cuda_func(name: str) -> Callable:
    def call_cuda(*args, **kwargs):
        # pylint: disable=import-outside-toplevel
        from ._backend import _C

        return getattr(_C, name)(*args, **kwargs)

    return call_cuda

initiate_cost = _make_lazy_cuda_func("initiate_cost")
propagation = _make_lazy_cuda_func("propagation")