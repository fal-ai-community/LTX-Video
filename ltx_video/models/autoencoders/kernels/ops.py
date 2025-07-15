import os

from torch.utils.cpp_extension import load

pixel_norm = load(
    name="pixel_norm",
    sources=[
        os.path.join(os.path.dirname(__file__), "pixel_norm.cpp"),
        os.path.join(os.path.dirname(__file__), "pixel_norm_cuda.cu"),
    ],
)


def pixel_norm_inplace(x, scale, shift, eps=1e-5):
    return pixel_norm.pixel_norm_inplace(x, scale, shift, eps)


inplace_add = load(
    name="inplace_add",
    sources=[
        os.path.join(os.path.dirname(__file__), "add_inplace.cpp"),
        os.path.join(os.path.dirname(__file__), "add_inplace_cuda.cu"),
    ],
)


def add_inplace(x, workspace, offset):
    return inplace_add.inplace_add(x, workspace, offset)
