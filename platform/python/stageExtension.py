# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Copy the built ``_native`` extension into the ``pypto_serving.platform`` package.

The module has to be importable as ``pypto_serving.platform._native``. ``pypto_serving`` is
a regular package, and a regular package always wins over an implicit namespace portion
regardless of ``sys.path`` order, so a build-tree mirror can never be imported once the
repository itself is importable. The extension must live inside the real package directory.

Stale extensions built for another interpreter ABI are removed, so switching
``-DpythonInterpreter`` cannot leave two ``.so`` files fighting over the same import.
"""

from __future__ import annotations

import filecmp
import shutil
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    """Stage ``argv[1]`` into directory ``argv[2]`` and touch the stamp file ``argv[3]``."""
    if len(argv) != 4:
        print("Usage: stageExtension.py <built-extension> <package-dir> <stamp>", file=sys.stderr)
        return 1

    source = Path(argv[1]).resolve()
    package_dir = Path(argv[2]).resolve()
    stamp = Path(argv[3])

    package_dir.mkdir(parents=True, exist_ok=True)
    destination = package_dir / source.name

    for stale in package_dir.glob("_native*.so"):
        if stale != destination:
            stale.unlink()

    # Compared by content, not by timestamp: an mtime guard cannot tell "already staged"
    # from "staged from the other build tree", and it refuses to re-stage a newer build that
    # happens to be older than what is sitting there.
    if not destination.exists() or not filecmp.cmp(source, destination, shallow=False):
        shutil.copy2(source, destination)

    stamp.write_text(str(destination) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
