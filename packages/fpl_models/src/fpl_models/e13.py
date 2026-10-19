"""E13: in-week price timing, from Wayback Machine captures of bootstrap-static.

FPL changes prices overnight from net transfers, so the same transfer made earlier or
later in the week can buy or sell at a different price. Gameweek-level history (E10)
can't see that; the Wayback Machine has ~2 captures a day of the live player list
for 2023-24 onwards.

    uv run python -m fpl_models.e13 download   # resumable, <= 1 request/s
    uv run python -m fpl_models.e13 extract    # -> data/silver/wayback/prices.parquet
    uv run python -m fpl_models.e13 analyse

Captures are reduced to the fields used here before being stored (gzip JSON), one a
day from mid-July 2023 to the end of May 2026.
"""

import argparse
import gzip
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

BOOTSTRAP = "https://fantasy.premierleague.com/api/bootstrap-static/"
BRONZE = Path("data/bronze/wayback/bootstrap_static")
CDX = Path("data/bronze/wayback/cdx.json")
SILVER = Path("data/silver/wayback/prices.parquet")
PAUSE_SECONDS = 3.0
START, END = datetime(2023, 7, 20, tzinfo=UTC), datetime(2026, 5, 31, tzinfo=UTC)
ELEMENT_FIELDS = (
    "id", "code", "element_type", "team", "now_cost", "cost_change_event",
    "transfers_in_event", "transfers_out_event", "selected_by_percent", "status",
)  # fmt: skip
EVENT_FIELDS = ("id", "deadline_time", "finished", "is_current", "is_next")


def parse_ts(ts: str) -> datetime:
    return datetime.strptime(ts, "%Y%m%d%H%M%S").replace(tzinfo=UTC)


def chosen_captures(cdx: list[list[str]]) -> list[str]:
    """One capture a day in the study period (skipping June): the first after 12:00 UTC,
    i.e. after that night's price changes, else the day's first. Prices change once a
    night, so a daily capture labels every change."""
    pick: dict[str, str] = {}
    for ts, *_ in sorted(cdx):
        when = parse_ts(ts)
        if START <= when <= END and when.month != 6:
            day = when.date().isoformat()
            if day not in pick or (parse_ts(pick[day]).hour < 12 <= when.hour):
                pick[day] = ts
    return sorted(pick.values())


def _body(raw: bytes) -> bytes:
    """Some captures were archived gzip-encoded."""
    return gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw


def reduce(payload: dict[str, Any]) -> dict[str, Any]:
    elements, events = payload["elements"], payload["events"]
    return {
        "elements": [{k: e.get(k) for k in ELEMENT_FIELDS} for e in elements],
        "events": [{k: e.get(k) for k in EVENT_FIELDS} for e in events],
    }


def _fetch(ts: str) -> str:
    url = f"https://web.archive.org/web/{ts}id_/{BOOTSTRAP}"
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                payload = json.loads(_body(response.read()))
            (BRONZE / f"{ts}.json.gz").write_bytes(
                gzip.compress(json.dumps(reduce(payload)).encode(), mtime=0)
            )
            time.sleep(PAUSE_SECONDS)
            return f"{ts} ok"
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError) as e:
            time.sleep(60 * (attempt + 1))  # refused: the archive's rate limit
            error = e
    return f"{ts} failed: {error}"


def download(workers: int = 1) -> None:
    """Sequential with a pause: the archive refuses connections above ~15 a minute."""
    cdx = json.loads(CDX.read_text())[1:]
    BRONZE.mkdir(parents=True, exist_ok=True)
    have_days = {p.name[:8] for p in BRONZE.glob("*.json.gz")}
    todo = [ts for ts in chosen_captures(cdx) if ts[:8] not in have_days]
    print(f"{len(todo)} captures to fetch", flush=True)
    with ThreadPoolExecutor(workers) as pool:
        for i, line in enumerate(pool.map(_fetch, todo)):
            if "failed" in line or i % 100 == 0:
                print(f"{i}/{len(todo)} {line}", flush=True)


def extract() -> pl.DataFrame:
    """One row per capture and player, with the gameweek whose transfer window is open."""
    rows = []
    for file in sorted(BRONZE.glob("*.json.gz")):
        ts = file.name.split(".")[0]
        data = json.loads(gzip.decompress(file.read_bytes()))
        when = parse_ts(ts)
        deadlines = sorted(
            (datetime.fromisoformat(e["deadline_time"].replace("Z", "+00:00")), e["id"])
            for e in data["events"]
            if e["deadline_time"]
        )
        upcoming = [(d, gw) for d, gw in deadlines if d > when]
        if not upcoming:
            continue
        next_deadline, next_gw = upcoming[0]
        previous = [d for d, _ in deadlines if d <= when]
        season_start = deadlines[0][0]
        season = f"{season_start.year}-{(season_start.year + 1) % 100:02d}"
        for e in data["elements"]:
            rows.append({
                "captured_at": when, "season": season, "next_gameweek": next_gw,
                "next_deadline": next_deadline,
                "previous_deadline": previous[-1] if previous else None,
                "element": e["id"], "player_code": e["code"], "position": e["element_type"],
                "price": e["now_cost"], "cost_change_event": e["cost_change_event"],
                "net_transfers": (e["transfers_in_event"] or 0) - (e["transfers_out_event"] or 0),
                "selected_by_percent": float(e["selected_by_percent"] or 0),
                "status": e["status"],
            })  # fmt: skip
    frame = pl.DataFrame(rows, infer_schema_length=None)
    SILVER.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(SILVER)
    print(f"{frame.height} rows, {frame['captured_at'].n_unique()} captures")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["download", "extract", "analyse"])
    args = parser.parse_args()
    if args.stage == "download":
        download()
    elif args.stage == "extract":
        extract()
    else:
        from fpl_models import e13_analysis  # noqa: PLC0415

        e13_analysis.main()


if __name__ == "__main__":
    main()
