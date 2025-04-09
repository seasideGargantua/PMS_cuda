from typing import Callable


def _make_lazy_cuda_func(name: str) -> Callable:
    def call_cuda(*args, **kwargs):
        # pylint: disable=import-outside-toplevel
        from ._backend import _C

        return getattr(_C, name)(*args, **kwargs)

    return call_cuda


rasterize_forward = _make_lazy_cuda_func("rasterize_forward")
rasterize_backward = _make_lazy_cuda_func("rasterize_backward")