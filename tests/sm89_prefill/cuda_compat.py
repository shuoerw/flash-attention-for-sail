"""Stage native PPU sources for the PPU SDK's CUDA-compatible PyTorch ABI.

Runtime names and architecture guards change in a generated tree; PPU
instructions, intrinsics, and source files remain intact.
"""
import re


_NAMES = {
    "acComplex": "cuComplex", "acFloatComplex": "cuFloatComplex",
    "acDoubleComplex": "cuDoubleComplex", "acCreal": "cuCreal",
    "acCrealf": "cuCrealf", "acCimag": "cuCimag", "acCimagf": "cuCimagf",
    "make_acFloatComplex": "make_cuFloatComplex", "make_acDoubleComplex": "make_cuDoubleComplex",
    "HGstream_st": "CUstream_st",
    "HGfunction": "CUfunction", "HGresult": "CUresult",
    "HGlaunchAttribute": "CUlaunchAttribute",
    "HGtensorMap": "CUtensorMap",
    "HGtensorMapDataType": "CUtensorMapDataType",
    "HGtensorMapInterleave": "CUtensorMapInterleave",
    "HGtensorMapSwizzle": "CUtensorMapSwizzle",
    "HGtensorMapL2promotion": "CUtensorMapL2promotion",
    "HGtensorMapFloatOOBfill": "CUtensorMapFloatOOBfill",
    "HGdeviceptr": "CUdeviceptr",
    "hgMemsetD32Async": "cuMemsetD32Async",
    "hgMemsetD16Async": "cuMemsetD16Async",
    "hgMemsetD8Async": "cuMemsetD8Async",
    "HGGC_SUCCESS": "CUDA_SUCCESS",
    "HGGC_ERROR_UNKNOWN": "CUDA_ERROR_UNKNOWN",
    "hgTensorMapEncodeTiled": "cuTensorMapEncodeTiled",
    "hgTensorMapEncodeIm2col": "cuTensorMapEncodeIm2col",
    "HGlaunchAttributeAD": "CUlaunchAttributeAD",
    "HGlaunchConfigAD": "CUlaunchConfigAD",
    "HGAD_LAUNCH_ATTRIBUTE_IGNORE": "CUAD_LAUNCH_ATTRIBUTE_IGNORE",
    "HGAD_LAUNCH_ATTRIBUTE_SCHED_PREFERENCE": "CUAD_LAUNCH_ATTRIBUTE_SCHED_PREFERENCE",
    "hgLaunchKernelExAD": "cuLaunchKernelExAD",
    "hgGetErrorName": "cuGetErrorName", "hgGetErrorString": "cuGetErrorString",
    "__hg_fp8_e4m3": "__nv_fp8_e4m3", "__hg_fp8_e5m2": "__nv_fp8_e5m2",
    "__hg_fp8_e8m0": "__nv_fp8_e8m0", "__hg_fp8x4_e4m3": "__nv_fp8x4_e4m3",
    "__ppu_bfloat16": "__nv_bfloat16", "__ppu_bfloat162": "__nv_bfloat162",
    "__ppu_bfloat16_raw": "__nv_bfloat16_raw",
    "__HGGC_ARCH__": "__CUDA_ARCH__",
    "hgToolsExt": "nvToolsExt",
}


def convert(text):
    def replace(match):
        name = match.group()
        if name in _NAMES:
            return _NAMES[name]
        for native, compat in (("hggcrt", "cudart"), ("HGGCRT", "CUDART"),
                               ("hggc", "cuda"), ("hgtx", "nvtx"), ("HGTX", "NVTX")):
            if name.startswith(native):
                return compat + name[len(native):]
        return name

    text = re.sub(r"\b[A-Za-z_][A-Za-z_0-9]*\b", replace, text)
    architectures = {"100": "800", "150": "890"}
    text = re.sub(
        r"(__CUDA_ARCH__\s*(?:==|!=|>=|<=|>|<)\s*)(100|150)\b",
        lambda match: match[1] + architectures[match[2]], text,
    )
    text = re.sub(
        r"\b(100|150)(\s*(?:==|!=|>=|<=|>|<)\s*__CUDA_ARCH__\b)",
        lambda match: architectures[match[1]] + match[2], text,
    )
    # CUDA headers already declare this type; native common.h forward-declares it.
    return re.sub(r"typedef struct CUstream_st\* cudaStream_t;\n", "", text)
