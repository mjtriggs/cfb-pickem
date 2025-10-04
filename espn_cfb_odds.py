#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
espn_cfb_odds.py

Pull weekly college-football odds from ESPN and report:
1) Top favourites (largest negative current spreads)
2) Highest totals (over/under)

By default, this queries FBS (groups=80) for a 7-day window starting today.

Usage:
    python espn_cfb_odds.py
    python espn_cfb_odds.py --days 3
    python espn_cfb_odds.py --start 2025-10-02 --end 2025-10-08
    python espn_cfb_odds.py --groups 80 81
    python espn_cfb_odds.py --topn 10
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests


def _iso_to_compact(date: dt.date) -> str:
    """
    Convert a date to ESPN's compact YYYYMMDD string.

    Args:
        date (datetime.date): A date object.

    Returns:
        str: The date formatted as YYYYMMDD.
    """
    return date.strftime("%Y%m%d")


def _build_dates_param(start: dt.date, end: dt.date) -> str:
    """
    Build ESPN `dates` query parameter.

    Args:
        start (datetime.date): Start date inclusive.
        end (datetime.date): End date inclusive.

    Returns:
        str: A dates range string like '20251001-20251007'.
    """
    return f"{_iso_to_compact(start)}-{_iso_to_compact(end)}"


def _get_scoreboard(
    start: dt.date, end: dt.date, groups: Iterable[int]
) -> Dict[str, Any]:
    """
    Query ESPN's NCAAF scoreboard endpoint for a date range and groups.

    Args:
        start (datetime.date): Start date inclusive.
        end (datetime.date): End date inclusive.
        groups (Iterable[int]): ESPN group IDs (e.g., [80] for FBS).

    Returns:
        dict: Parsed JSON payload from ESPN.

    Raises:
        requests.HTTPError: If the HTTP request fails.
    """
    base = "https://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard"
    params = {
        "dates": _build_dates_param(start, end),
        "groups": ",".join(str(g) for g in groups) if groups else "80",
    }
    try:
        resp = requests.get(base, params=params, timeout=20)
        resp.raise_for_status()
        return resp.json()
    except requests.HTTPError:
        # Fallback: query per group and merge events.
        all_events: List[Dict[str, Any]] = []
        for g in groups:
            r = requests.get(base, params={"dates": params["dates"], "groups": g}, timeout=20)
            r.raise_for_status()
            j = r.json()
            all_events.extend(j.get("events", []))
        return {"events": all_events}


def _safe_float(x: Optional[str | float | int]) -> Optional[float]:
    """
    Convert a value to float safely.

    Args:
        x (Optional[Union[str, float, int]]): Input value.

    Returns:
        Optional[float]: Float value or None if not parseable.
    """
    if x is None:
        return None
    try:
        s = str(x).strip().replace("½", ".5")
        return float(s)
    except Exception:
        return None


def _extract_odds_from_event(event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Extract teams, favourite & spread, and total from an ESPN event JSON.

    This version is more assertive about identifying the favourite:
    1) Prefer explicit home/away spreads (choose the team with the negative spread).
    2) If only one side has a spread:
       - negative => that side is favourite
       - positive => the *other* side is favourite (implied).
    3) If both sides have positive or both missing, fall back to parsing the `details`
       string (e.g., "Georgia -19.5") and map that to a team name.
    4) As a last resort, keep favourite as None.

    Returns:
        Optional[dict]: {
            "event_id": str,
            "kickoff": str,
            "home_team": str,
            "away_team": str,
            "favourite": Optional[str],
            "spread": Optional[float],   # negative for favourite if available
            "total": Optional[float],
        }
    """
    comp = (event.get("competitions") or [None])[0]
    if not comp:
        return None

    competitors = comp.get("competitors") or []
    if len(competitors) != 2:
        return None

    home = next((c for c in competitors if c.get("homeAway") == "home"), competitors[0])
    away = next((c for c in competitors if c.get("homeAway") == "away"), competitors[-1])

    def _team_name(c: Dict[str, Any]) -> str:
        return (
            c.get("team", {}).get("displayName")
            or c.get("team", {}).get("name")
            or c.get("team", {}).get("shortDisplayName")
            or "Unknown"
        )

    home_team = _team_name(home)
    away_team = _team_name(away)
    kickoff = comp.get("date") or event.get("date")

    odds_list = comp.get("odds") or event.get("odds") or []

    spread_value: Optional[float] = None
    favourite_name: Optional[str] = None
    total_value: Optional[float] = None

    # --- helper to try explicit home/away spreads first
    def infer_from_home_away_spreads(od: Dict[str, Any]) -> Tuple[Optional[str], Optional[float]]:
        h = od.get("homeTeamOdds") or {}
        a = od.get("awayTeamOdds") or {}
        h_sp = _safe_float(h.get("spread"))
        a_sp = _safe_float(a.get("spread"))

        # Typical convention: favourite has negative spread
        if h_sp is not None and a_sp is not None:
            if h_sp < 0 and a_sp > 0:
                return home_team, h_sp
            if a_sp < 0 and h_sp > 0:
                return away_team, a_sp
            # If both negative or both positive (rare / provider artefact),
            # choose the more negative one.
            if h_sp < a_sp:
                return home_team, h_sp
            if a_sp < h_sp:
                return away_team, a_sp
            # Equal spreads gives no signal
            return None, None

        # If only one side has a spread:
        if h_sp is not None and a_sp is None:
            if h_sp < 0:
                return home_team, h_sp
            # positive home spread implies away favourite with the negative mirror
            return away_team, -abs(h_sp)
        if a_sp is not None and h_sp is None:
            if a_sp < 0:
                return away_team, a_sp
            return home_team, -abs(a_sp)

        return None, None

    # --- helper to parse 'details', e.g., "Georgia -19.5"
    def infer_from_details(od: Dict[str, Any]) -> Tuple[Optional[str], Optional[float]]:
        details = (od.get("details") or "").strip()
        if not details:
            return None, None
        parts = details.rsplit(" ", 1)
        if len(parts) != 2:
            return None, None
        maybe_team, maybe_num = parts
        sp = _safe_float(maybe_num)
        if sp is None:
            return None, None

        # Try to map the text team to home/away names with simple normalisation
        def norm(s: str) -> str:
            return "".join(ch.lower() for ch in s if ch.isalnum())

        mt = norm(maybe_team)
        ht = norm(home_team)
        at = norm(away_team)

        mapped: Optional[str] = None
        if mt and (mt in ht or ht in mt):
            mapped = home_team
        elif mt and (mt in at or at in mt):
            mapped = away_team
        else:
            # Could not confidently map; still return the raw team name
            mapped = maybe_team

        # Ensure favourite spread is negative
        sp_neg = -abs(sp)
        return mapped, sp_neg

    # Iterate odds entries; prefer explicit home/away spreads, then details; also pick up total.
    for od in odds_list:
        if total_value is None:
            total_value = _safe_float(od.get("overUnder")) or total_value

        if favourite_name is None or spread_value is None:
            fav1, sp1 = infer_from_home_away_spreads(od)
            if fav1 is not None and sp1 is not None:
                favourite_name, spread_value = fav1, sp1
                continue  # still allow later odds to supply total if missing

        if (favourite_name is None or spread_value is None) and od.get("details"):
            fav2, sp2 = infer_from_details(od)
            if fav2 is not None and sp2 is not None:
                favourite_name, spread_value = fav2, sp2

    # Final fallback for total
    if total_value is None:
        total_value = _safe_float(comp.get("overUnder"))

    return {
        "event_id": str(event.get("id")),
        "kickoff": kickoff,
        "home_team": home_team,
        "away_team": away_team,
        "favourite": favourite_name,
        "spread": spread_value,
        "total": total_value,
    }


def _gather_odds(
    start: dt.date, end: dt.date, groups: Iterable[int]
) -> List[Dict[str, Any]]:
    """
    Pull and parse odds for all events in the range/groups.

    Args:
        start (datetime.date): Start date inclusive.
        end (datetime.date): End date inclusive.
        groups (Iterable[int]): ESPN groups to include.

    Returns:
        List[dict]: List of parsed odds entries per event.
    """
    data = _get_scoreboard(start, end, groups)
    out: List[Dict[str, Any]] = []
    for ev in data.get("events", []):
        parsed = _extract_odds_from_event(ev)
        if parsed:
            out.append(parsed)
    return out


def _rank_top_favourites(
    rows: List[Dict[str, Any]], topn: int
) -> List[Dict[str, Any]]:
    """
    Rank by most negative spread (largest absolute favourite).

    Args:
        rows (List[dict]): Parsed odds rows.
        topn (int): Number of rows to return.

    Returns:
        List[dict]: Top-N favourites with spread present.
    """
    with_spread = [r for r in rows if isinstance(r.get("spread"), (int, float))]
    with_spread.sort(key=lambda r: r["spread"])  # most negative first
    return with_spread[:topn]


def _rank_highest_totals(
    rows: List[Dict[str, Any]], topn: int
) -> List[Dict[str, Any]]:
    """
    Rank by highest totals (over/under).

    Args:
        rows (List[dict]): Parsed odds rows.
        topn (int): Number of rows to return.

    Returns:
        List[dict]: Top-N rows by total descending.
    """
    with_total = [r for r in rows if isinstance(r.get("total"), (int, float))]
    with_total.sort(key=lambda r: r["total"], reverse=True)
    return with_total[:topn]


def _fmt_match(r: Dict[str, Any]) -> str:
    """
    Format a single row for console output.

    Args:
        r (dict): A parsed odds row.

    Returns:
        str: Human-readable summary.
    """
    fav = r.get("favourite") or "Unknown favourite"
    sp = r.get("spread")
    tot = r.get("total")
    kickoff = r.get("kickoff") or "TBD"
    return (
        f"{r['away_team']} at {r['home_team']} | kickoff: {kickoff} | "
        f"fav: {fav} spread: {sp if sp is not None else 'NA'} | "
        f"total: {tot if tot is not None else 'NA'}"
    )


def main(argv: Optional[List[str]] = None) -> int:
    """
    CLI entry-point. Fetch odds, then print:
    - Top favourites (largest negative spreads)
    - Highest totals

    Args:
        argv (Optional[List[str]]): CLI args (for testing). Defaults to sys.argv[1:].

    Returns:
        int: Process exit code (0 = success).
    """
    parser = argparse.ArgumentParser(
        description="Extract college-football betting odds from ESPN and identify weekly top favourites & highest totals."
    )
    today = dt.date.today()
    parser.add_argument(
        "--start",
        type=str,
        default=today.isoformat(),
        help="Start date (YYYY-MM-DD). Default: today.",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=(today + dt.timedelta(days=6)).isoformat(),
        help="End date (YYYY-MM-DD). Default: today+6.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Shortcut: number of days from start (overrides --end).",
    )
    parser.add_argument(
        "--groups",
        type=int,
        nargs="+",
        default=[80],  # 80 ≈ FBS
        help="ESPN group IDs (e.g., 80=FBS). Pass multiple to broaden coverage.",
    )
    parser.add_argument(
        "--topn", type=int, default=8, help="How many to show for each table."
    )

    args = parser.parse_args(argv)

    start_date = dt.date.fromisoformat(args.start)
    end_date = (
        start_date + dt.timedelta(days=args.days)
        if args.days is not None
        else dt.date.fromisoformat(args.end)
    )

    rows = _gather_odds(start_date, end_date, args.groups)

    if not rows:
        print("No events or odds found in the specified window.", file=sys.stderr)
        return 1

    top_favs = _rank_top_favourites(rows, args.topn)
    top_totals = _rank_highest_totals(rows, args.topn)

    print("\n=== Top Favourites (largest negative spreads) ===")
    for r in top_favs:
        print(_fmt_match(r))

    print("\n=== Highest Totals (over/under) ===")
    for r in top_totals:
        print(_fmt_match(r))

    return 0 if (top_favs and top_totals) else 2


if __name__ == "__main__":
    raise SystemExit(main())
