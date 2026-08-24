from __future__ import annotations

import argparse
import calendar
import csv
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_START = "2020-01"
DEFAULT_END = "2023-12"
DEFAULT_OUT = Path("training/vendor/dukascopy_xau_m1/xauusd/bid/m1")


@dataclass(frozen=True)
class Month:
    year: int
    month: int

    @property
    def label(self) -> str:
        return f"{self.year:04d}-{self.month:02d}"

    @property
    def start(self) -> str:
        return f"{self.year:04d}-{self.month:02d}-01"

    @property
    def next_start(self) -> str:
        if self.month == 12:
            return f"{self.year + 1:04d}-01-01"
        return f"{self.year:04d}-{self.month + 1:02d}-01"

    @property
    def filename(self) -> str:
        return f"xauusd_bid_m1_{self.year:04d}_{self.month:02d}.csv"


def parse_month(value: str) -> Month:
    try:
        year_s, month_s = value.split("-")
        year = int(year_s)
        month = int(month_s)
    except Exception as exc:
        raise argparse.ArgumentTypeError("month must be YYYY-MM") from exc
    if not 1 <= month <= 12:
        raise argparse.ArgumentTypeError("month must be YYYY-MM")
    return Month(year, month)


def months_between(start: Month, end: Month) -> list[Month]:
    out: list[Month] = []
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        out.append(Month(y, m))
        if m == 12:
            y, m = y + 1, 1
        else:
            m += 1
    return out


def validate_csv(path: Path, month: Month) -> dict[str, object]:
    rows = 0
    first_ts = None
    last_ts = None
    prev_ts = None
    duplicates = 0
    bad_ohlc = 0
    bad_interval = 0

    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        required = ["timestamp", "open", "high", "low", "close"]
        if reader.fieldnames != required:
            raise RuntimeError(
                f"unexpected columns in {path}: {reader.fieldnames}, expected {required}"
            )

        for row in reader:
            rows += 1
            ts = int(row["timestamp"])
            o = float(row["open"])
            h = float(row["high"])
            l = float(row["low"])
            c = float(row["close"])

            if first_ts is None:
                first_ts = ts
            last_ts = ts

            if prev_ts is not None:
                if ts == prev_ts:
                    duplicates += 1
                elif ts < prev_ts:
                    raise RuntimeError(f"timestamps not sorted in {path}")
                elif ts - prev_ts != 60_000:
                    bad_interval += 1
            prev_ts = ts

            if min(o, h, l, c) <= 0 or h < max(o, l, c) or l > min(o, h, c):
                bad_ohlc += 1

    if rows == 0:
        raise RuntimeError(f"empty csv: {path}")

    start_ms = int(
        datetime(month.year, month.month, 1, tzinfo=timezone.utc).timestamp() * 1000
    )
    if first_ts is None or first_ts < start_ms:
        raise RuntimeError(f"unexpected first timestamp in {path}: {first_ts}")

    return {
        "month": month.label,
        "file": str(path),
        "rows": rows,
        "first_timestamp_ms": first_ts,
        "last_timestamp_ms": last_ts,
        "duplicates": duplicates,
        "non_1m_gaps": bad_interval,
        "bad_ohlc": bad_ohlc,
        "size_bytes": path.stat().st_size,
    }


def run_download(month: Month, destination: Path) -> dict[str, object]:
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        try:
            result = validate_csv(destination, month)
            result["status"] = "cached"
            return result
        except Exception:
            destination.unlink()

    with tempfile.TemporaryDirectory(prefix="ten_duka_") as td:
        work = Path(td)
        cmd = [
            "npx",
            "-y",
            "dukascopy-node@latest",
            "-i",
            "xauusd",
            "-from",
            month.start,
            "-to",
            month.next_start,
            "-t",
            "m1",
            "-f",
            "csv",
        ]

        print("RUN:", " ".join(cmd), flush=True)
        proc = subprocess.run(
            cmd,
            cwd=work,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        print(proc.stdout, end="", flush=True)

        if proc.returncode != 0:
            raise RuntimeError(f"dukascopy-node failed for {month.label}")

        files = list((work / "download").glob("*.csv"))
        if len(files) != 1:
            raise RuntimeError(
                f"expected exactly one csv for {month.label}, found {len(files)}"
            )

        shutil.move(str(files[0]), destination)

    result = validate_csv(destination, month)
    result["status"] = "downloaded"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=parse_month, default=parse_month(DEFAULT_START))
    parser.add_argument("--end", type=parse_month, default=parse_month(DEFAULT_END))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    if (args.start.year, args.start.month) > (args.end.year, args.end.month):
        raise SystemExit("--start must be <= --end")

    months = months_between(args.start, args.end)
    if args.limit > 0:
        months = months[: args.limit]

    print("TEN Dukascopy XAUUSD bid M1 downloader")
    print("months:", len(months))
    print("out:", args.out)
    print()

    manifest: list[dict[str, object]] = []
    for i, month in enumerate(months, 1):
        print(f"[{i}/{len(months)}] {month.label}")
        result = run_download(month, args.out / month.filename)
        manifest.append(result)
        print(json.dumps(result, indent=2))
        print()

    manifest_path = args.out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print("DONE")
    print("manifest:", manifest_path)
    print("files:", len(manifest))
    print("rows:", sum(int(x["rows"]) for x in manifest))
    print("bytes:", sum(int(x["size_bytes"]) for x in manifest))


if __name__ == "__main__":
    main()
