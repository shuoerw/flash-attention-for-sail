"""Compare emitted PPU kernel instructions across builds, ignoring symbol names."""
import argparse
import hashlib
import json
from pathlib import Path
import struct


def kernels(directory):
    objects = list((directory / "keep").glob("*-llc.o"))
    if len(objects) != 1:
        raise ValueError(f"Expected one PPU object in {directory}, found {len(objects)}")
    data = objects[0].read_bytes()
    if data[:6] != b"\x7fELF\x02\x01":
        raise ValueError(f"Expected a little-endian ELF64 object: {objects[0]}")
    offset = struct.unpack_from("<Q", data, 40)[0]
    size, count, index = struct.unpack_from("<HHH", data, 58)
    headers = [struct.unpack_from("<IIQQQQIIQQ", data, offset + i * size) for i in range(count)]
    header = headers[index]
    strings = data[header[4]:header[4] + header[5]]
    result = {}
    for header in headers:
        name = strings[header[0]:strings.find(b"\0", header[0])].decode()
        if name.startswith(".text.kernel."):
            result[name] = dict(bytes=header[5], sha256=hashlib.sha256(
                data[header[4]:header[4] + header[5]]).hexdigest())
    if not result:
        raise ValueError(f"No device kernels in {objects[0]}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shared-only", action="store_true",
                        help="Check all reference kernels by name; allow additional candidate specializations")
    args = parser.parse_args()
    reference = kernels(args.reference)
    candidate = kernels(args.candidate)
    rows = []
    if args.shared_only:
        missing = reference.keys() - candidate.keys()
        if missing:
            raise ValueError(f"Reference kernels missing from candidate: {sorted(missing)}")
        candidate = {name: candidate[name] for name in reference}
    for name, instructions in candidate.items():
        matches = [key for key, value in reference.items()
                   if instructions == value and (not args.shared_only or key == name)]
        rows.append(dict(kernel=name, instructions=instructions, identical_to=matches))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n")
    failures = [row["kernel"] for row in rows if not row["identical_to"]]
    print(f"{len(rows) - len(failures)}/{len(rows)} kernel bodies match the reference")
    for name in failures:
        print("CHANGED", name)
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
