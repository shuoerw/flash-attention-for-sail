"""Build a standalone SM89 prefill test library without installing a package."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sm89_prefill"))
from cuda_compat import convert


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--actlize", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compiler = shutil.which("nvcc")
    if compiler is None:
        parser.error("Add the PPU SDK CUDA_SDK/bin directory to PATH")
    args.output.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for source, destination in [(args.source / "hopper", args.output / "hopper"),
                                (args.actlize / "include", args.output / "include")]:
        for path in source.rglob("*"):
            if not path.is_file() or path.name.startswith("._"):
                continue
            if path.suffix not in (".h", ".hpp", ".cuh", ".inl"):
                continue
            relative = path.relative_to(source).as_posix()
            target = destination / convert(relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(convert(path.read_text()))
            if source.name == "hopper":
                hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    scheduler = args.source / "hopper/flash_prepare_scheduler.cu"
    (args.output / "hopper/flash_prepare_scheduler.cu").write_text(convert(scheduler.read_text()))
    runner = Path(__file__).with_name("runner.cu")
    shutil.copyfile(runner, args.output / "runner.cu")
    hashes["flash_prepare_scheduler.cu"] = hashlib.sha256(scheduler.read_bytes()).hexdigest()
    hashes["runner.cu"] = hashlib.sha256(runner.read_bytes()).hexdigest()
    keep = args.output / "keep"
    keep.mkdir(exist_ok=True)
    command = [compiler, "-shared", "-Xcompiler", "-fPIC", "-O3", "-std=c++17",
               "--use_fast_math", "-arch=sm_89", "--resource-usage", "-lineinfo",
               "--expt-relaxed-constexpr", "--expt-extended-lambda", "-DUSE_PPU",
               "-DUSE_AIU=1", "-DFLASHATTENTION_DISABLE_SM90", "-DFLASHATTENTION_DISABLE_FP8",
               "-DFLASHATTENTION_DISABLE_BACKWARD", "-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED",
               "-DCUTLASS_ENABLE_GDC_FOR_SM90", "-DNDEBUG",
               "-I" + str(args.output / "hopper"), "-I" + str(args.output / "include"),
               str(args.output / "runner.cu"), "-o", str(args.output / "libfa.so"),
               "-keep", "-keep-dir", str(keep)]
    for flag in ["-ppu-max-vreg-count=256", "-ppu-sink-matrix-addr=true",
                 "-ppu-max-alloca-byte-size=320", "-ppu-sink-async-addr=true",
                 "-ppu-sink-load-addr=true", "-ppu-sink-store-addr=true",
                 "-ppu-alloca-half-ldst-simplify=true", "-ppu-volatile-yield=false",
                 "-sort-copy-before-coalesce=true"]:
        command.extend(["-mllvm", flag])
    result = subprocess.run(command, check=False)
    record = dict(command=command, returncode=result.returncode, source_sha256=hashes)
    if result.returncode == 0:
        record["library_sha256"] = hashlib.sha256((args.output / "libfa.so").read_bytes()).hexdigest()
        contract_source = Path(__file__).with_name("contract.cu")
        shutil.copyfile(contract_source, args.output / "contract.cu")
        hashes["contract.cu"] = hashlib.sha256(contract_source.read_bytes()).hexdigest()
        contract_command = [compiler, "-O2", "-std=c++17", "-arch=sm_89",
                            "--expt-relaxed-constexpr", "--expt-extended-lambda",
                            "-DUSE_PPU", "-DUSE_AIU=1", "-DFLASHATTENTION_DISABLE_SM90",
                            "-DFLASHATTENTION_DISABLE_FP8",
                            "-I" + str(args.output / "hopper"), "-I" + str(args.output / "include"),
                            str(args.output / "contract.cu"), "-o", str(args.output / "contract")]
        contract_result = subprocess.run(contract_command, check=False)
        record["contract_compile_returncode"] = contract_result.returncode
        if contract_result.returncode == 0:
            contract_result = subprocess.run([str((args.output / "contract").resolve())], check=False)
            record["contract_test_returncode"] = contract_result.returncode
        if contract_result.returncode != 0:
            result = contract_result
    (args.output / "build.json").write_text(json.dumps(record, indent=2) + "\n")
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
