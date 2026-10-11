"""Inspect the immutable upstream prior archive without deserializing its models."""

import argparse
import hashlib
import json
import tarfile
import urllib.request
from pathlib import Path, PurePosixPath

REVISION = "5730e2d4559091657c3cd3fe248ced49efb1607f"
DIGEST = "b3b4d9e34ce41f96a92420fadb5addb5f62ceaf2f64d99f8afa9acf955498fa6"
SIZE = 18818365440
URL = f"https://huggingface.co/3DAIGC/LHM/resolve/{REVISION}/LHM_prior_model.tar"


class HashingReader:
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()
        self.size = 0

    def read(self, size=-1):
        block = self.stream.read(size)
        self.digest.update(block)
        self.size += len(block)
        return block


def inspect_archive(output):
    files = []
    paths = set()
    with urllib.request.urlopen(URL, timeout=120) as response:
        reader = HashingReader(response)
        with tarfile.open(fileobj=reader, mode="r|", bufsize=1024 * 1024) as archive:
            for member in archive:
                path = PurePosixPath(member.name)
                if path.is_absolute() or ".." in path.parts or str(path) in paths:
                    raise ValueError(f"Unsafe or duplicate member path: {member.name}")
                paths.add(str(path))
                if member.isdir():
                    continue
                if not member.isfile():
                    raise ValueError(f"Unsupported archive member: {member.name}")
                digest = hashlib.sha256()
                with archive.extractfile(member) as stream:
                    while block := stream.read(1024 * 1024):
                        digest.update(block)
                entry = {"path": str(path), "sha256": digest.hexdigest(), "size": member.size}
                files.append(entry)
                print(json.dumps(entry), flush=True)
        while reader.read(1024 * 1024):
            pass
        actual = reader.digest.hexdigest()
        if actual != DIGEST or reader.size != SIZE:
            raise ValueError(f"Archive integrity failure: {actual}, {reader.size} bytes")
    result = {"source": URL, "archive_sha256": actual, "archive_size": reader.size, "files": files}
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Verified archive {actual} ({reader.size} bytes)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    inspect_archive(parser.parse_args().output)
