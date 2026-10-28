"""Run the MiMo correctness and performance regression matrix on one idle PPU."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--groups", nargs="+")
    parser.add_argument("--previous-library", type=Path,
                        help="Also require bit-exact agreement with a previously validated library.")
    parser.add_argument("--optimized-aa", action="store_true",
                        help="Include an identical optimized-library copy to measure optimized timing variation.")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    libraries = [args.root / name / "libfa.so" for name in ("control", "control-copy", "release")]
    libraries[1].parent.mkdir(exist_ok=True)
    shutil.copy2(libraries[0], libraries[1])
    if args.previous_library:
        previous = args.root / "previous/libfa.so"
        previous.parent.mkdir(exist_ok=True)
        shutil.copy2(args.previous_library, previous)
        libraries.append(previous)
    if args.optimized_aa:
        copy = args.root / "release-copy/libfa.so"
        copy.parent.mkdir(exist_ok=True)
        shutil.copy2(args.root / "release/libfa.so", copy)
        libraries.append(copy)
    (args.output / "hashes.json").write_text(json.dumps({
        path.parent.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in libraries
    }, indent=2) + "\n")
    suites = [
        ("contexts-4k", [[1, 4096, 4096 * 2**i] for i in range(9)], [], 51),
        ("contexts-64k", [[1, min(k, 65536), k] for k in [8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]], [], 7),
        ("contexts-repeat", [[1, min(k, 65536), k] for k in [8192, 16384, 32768, 65536, 131072, 262144]], [], 31),
        ("mixed-64k", [], ["--query-lengths", "4096", "8192", "32768", "20480",
                              "--kv-lengths", "4096", "8192", "98304", "151552", "--compact-pages"], 51),
        ("mixed-64k-1m", [], ["--query-lengths", "4096", "8192", "32768", "20480",
                                 "--kv-lengths", "4096", "8192", "524288", "1048576", "--compact-pages"], 11),
        ("mixed-64k-split", [], ["--query-lengths", "4096", "8192", "32768", "20480",
                                    "--kv-lengths", "4096", "8192", "98304", "151552", "--compact-pages",
                                    "--splits", "16", "--reference-splits", "1"], 21),
        ("extended-bounds", [[1, 65537, 1048640], [257, 256, 4096]], ["--compact-pages"], 9),
        ("short", [[1, q, q] for q in [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]]
         + [[4, 512, 512], [8, 256, 256], [2, 1024, 1024], [4, 4096, 4096], [2, 6144, 6144]], [], 51),
        ("boundary", [[1, 4096, k] for k in [32704, 32768, 32769, 65535, 65536, 65539]], [], 31),
        ("prefix", [[b, q, k] for b, q in [(1, 512), (1, 4096), (4, 2048)]
                    for k in [131072, 262144, 1048576]], [], 21),
        ("mixed-ramp", [[4, 1025, 65539], [4, 2048, 262144], [8, 512, 1048576]],
         ["--kv-pattern", "ramp", "--kv-short-length", "4096", "--q-pattern", "ramp"], 21),
        ("mixed-one-long", [[4, 1025, 65539], [4, 2048, 262144], [16, 512, 1048576]],
         ["--kv-pattern", "one-long", "--kv-short-length", "4096", "--q-pattern", "one-short"], 21),
        ("large", [[1, 65536, 65536], [1, 65536, 131072], [1, 32768, 1048576], [1, 65536, 1048576]], [], 5),
        ("split16", [[1, 4096, 131072], [1, 4096, 1048576], [2, 1025, 66001]],
         ["--splits", "16", "--reference-splits", "1"], 21),
        ("local", [[1, 1, 4096], [96, 4, 8192], [16, 4, 1048576], [4, 512, 131072]], ["--windows", "128"], 51),
        ("decode", [[1, 1, 4096], [96, 4, 8192], [16, 4, 1048576]], [], 51),
        ("query-boundary", [[1, q, 513] for q in [464, 465, 466, 467, 468, 469]]
         + [[8, 63, 63], [8, 65, 129]], [], 51),
        ("dispatch-boundary", [[1, q, 4096] for q in [464, 465, 466, 467, 468, 469]]
         + [[1, 512, k] for k in [960, 1023, 1024, 1025]], [], 51),
        ("sinks", [[1, 512, 577], [4, 1025, 65539], [1, 512, 1048576]], ["--attention-sinks"], 21),
        ("compact-pages", [[4, 1025, 65539], [8, 512, 1048576]],
         ["--compact-pages", "--kv-pattern", "one-long", "--kv-short-length", "4096", "--q-pattern", "ramp"], 21),
        ("graph-replay", [[4, 512, 65536]], ["--graph-replay-stress", "--kv-short-length", "63"], 21),
        ("graph-empty", [[4, 512, 65536]], ["--graph-replay-stress", "--kv-short-length", "0"], 21),
        ("empty-pages", [], ["--correctness-only", "--native-reference-only", "--empty-sequences", "--capacity-k", "65536", "--compact-pages"], 1),
        ("partial-pages", [], ["--correctness-only", "--partial-pages", "--splits", "16", "--reference-splits", "1"], 1),
        ("large-batch", [[32, 17, 4096], [64, 65, 4097], [128, 256, 8192], [256, 256, 8192]], [], 31),
        ("large-batch-uneven", [[64, 257, 65539], [128, 129, 1048576], [256, 65, 1048576]],
         ["--compact-pages", "--kv-pattern", "one-long", "--kv-short-length", "4096", "--q-pattern", "ramp"], 31),
        ("short-repeat", [[1, q, q] for q in [128, 256, 512, 1024, 2048]]
         + [[4, 512, 512], [8, 256, 256], [1, 1, 4096], [96, 4, 8192]], [], 301),
    ]
    records = []
    for label, cases, extra, repeats in suites:
        if args.groups and label not in args.groups:
            continue
        output = args.output / (label + ".json")
        command = [sys.executable, str(Path(__file__).with_name("replay.py")),
                   "--libraries", *map(str, libraries), "--only-libraries", "--shared-workspace",
                   "--compare-baseline", "control", "--exact-groups", "control,control-copy",
                   "--repeats", str(repeats), "--seed", "1515", "--windows", "-1",
                   "--output", str(output), *extra]
        for case in cases:
            command.extend(["--case", *map(str, case)])
        with (args.output / (label + ".log")).open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=3600)
        records.append(dict(group=label, command=command, returncode=result.returncode))
        (args.output / "runs.json").write_text(json.dumps(records, indent=2) + "\n")
        if result.returncode:
            raise SystemExit(result.returncode)
        for row in json.loads(output.read_text())["benchmarks"]:
            times = row["median_us"]
            print(label, row["batch"], row["sq"], row["sk"],
                  {name: round((time / times["control"] - 1) * 100, 3) for name, time in times.items()}, flush=True)


if __name__ == "__main__":
    main()
