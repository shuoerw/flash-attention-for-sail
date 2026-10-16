"""Compile unaffected template paths using a completed standalone build's flags."""
import argparse
import json
from pathlib import Path
import subprocess

from cuda_compat import convert

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--build", type=Path, required=True)
args = parser.parse_args()
command = json.loads((args.build / "build.json").read_text())["command"]
source = args.build / "compile_coverage.cu"
source.write_text(convert(Path(__file__).with_name("compile_coverage.cu").read_text()))
command[command.index(str(args.build / "runner.cu"))] = str(source)
command[command.index("-o") + 1] = str(args.build / "compile_coverage.so")
keep = args.build / "coverage-keep"
keep.mkdir(exist_ok=True)
command[command.index("-keep-dir") + 1] = str(keep)
result = subprocess.run(command, check=False)
(args.build / "compile_coverage.json").write_text(
    json.dumps(dict(command=command, returncode=result.returncode), indent=2) + "\n")
raise SystemExit(result.returncode)
