"""Parser for Sea-Bird SBE37 CNV files.

Integration:
- Use in SBE37 notebooks/scripts to read CNV input into dataframe.
- Returns parsed columns and computed ``datetime`` from ``Instrument Time``.

Copy-paste:
    from tools.parsers.read_sbe37 import read_sbe37_cnv

    df = read_sbe37_cnv(cfg["input_file"], verbose=True)
"""

from __future__ import annotations

from pathlib import Path
import re

import pandas as pd

from tools.helpers import clean_label


def read_sbe37_cnv(input_file: str | Path, verbose: bool = True) -> pd.DataFrame:
    """Read an SBE37 CNV file and return a dataframe with a computed datetime column."""
    input_path = Path(input_file)

    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    col_names = []
    start = None
    start_time_str = None

    for i, line in enumerate(lines):
        s = line.strip()
        if s == "*END*":
            start = i + 1

        # Example: # name 1 = tv290C: Temperature [ITS-90, deg C]
        m = re.match(r"#\s*name\s+\d+\s*=\s*([^:]+):\s*(.+)$", s)
        if m:
            col_names.append(clean_label(m.group(1).strip(), m.group(2).strip()))

        # Example: # start_time = Dec 06 2019 19:45:00 [Instrument's time stamp, first scan]
        t = re.match(r"#\s*start_time\s*=\s*(.+?)\s*\[", s)
        if t:
            start_time_str = t.group(1).strip()

    if start is None:
        raise ValueError("Could not find '*END*' in CNV header.")
    if not start_time_str:
        raise ValueError("Could not find '# start_time =' in CNV header.")

    df = pd.read_csv(
        input_path,
        skiprows=start,
        sep=r"\s+",
        header=None,
        names=col_names if col_names else None,
        engine="python",
    )

    # Start at header start_time, then advance by elapsed instrument time.
    df_start = pd.to_datetime(start_time_str, format="%b %d %Y %H:%M:%S")
    elapsed_days = df["Instrument Time"] - df["Instrument Time"].iloc[0]
    df["datetime"] = df_start + pd.to_timedelta(elapsed_days, unit="D")

    if verbose:
        print(f"Start datetime used: {df_start}")
        print(df.columns.tolist())
        # Check (head/tail)
        print(df)

    return df


__all__ = ["read_sbe37_cnv"]
