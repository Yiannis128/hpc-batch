#!/usr/bin/env python3
"""Fail unless the built wheel carries the data files the installer needs.

The wheel did not carry the systemd unit until the installer needed it: the
package looked fine and left you with no way to run the service. Run from
both CI and the release workflow, so the two cannot disagree about what a
correctly packaged wheel contains.
"""

import glob
import sys
import zipfile

REQUIRED = ("hpc_batch/data/hpc-batch.service",)


def main() -> None:
    wheels = glob.glob("dist/*.whl")
    if not wheels:
        sys.exit("no wheel in dist/ to check")
    wheel = wheels[0]
    names = zipfile.ZipFile(wheel).namelist()
    missing = [name for name in REQUIRED if name not in names]
    if missing:
        sys.exit(f"{wheel} is missing {', '.join(missing)}")
    print(f"{wheel} carries {', '.join(REQUIRED)}")


if __name__ == "__main__":
    main()
