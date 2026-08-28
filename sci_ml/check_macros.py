"""Fail loudly on a result macro that is quoted but never generated.

Every headline number in the paper is a \\newcommand written by make_tables.py.
An undefined one is not a compile error in LaTeX -- it is, but the message is
buried -- and a stale one silently keeps an old value. Run this before building.
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREFIXES = ("Abl", "All", "Pinn", "Cheb", "FD", "Frontier", "RefSelf")
USE = re.compile(r"\\(" + "|".join(PREFIXES) + r")([A-Za-z]+)")
DEF = re.compile(r"newcommand\{\\([A-Za-z]+)\}")


def main():
    numbers = HERE / "tables" / "numbers.tex"
    if not numbers.exists():
        print(f"missing {numbers}; run make_tables.py first", file=sys.stderr)
        return 1
    defined = set(DEF.findall(numbers.read_text(encoding="utf-8")))

    used = set()
    for path in HERE.glob("*.tex"):
        if path.name == "checklist.tex":
            # the checklist quotes result macros too
            pass
        text = path.read_text(encoding="utf-8", errors="replace")
        used |= {a + b for a, b in USE.findall(text)}

    missing = sorted(used - defined)
    unused = sorted(defined - used)
    if missing:
        print("USED BUT NOT GENERATED:", ", ".join(missing), file=sys.stderr)
    if unused:
        print("generated but unused  :", ", ".join(unused))
    if not missing:
        print(f"all {len(used)} result macros resolve")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
