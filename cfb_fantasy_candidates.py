#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
cfb_fantasy_candidates.py

Pull NCAA football betting lines (ESPN) and season leaders for Passing, Rushing,
Receiving, and Returning. Compute fantasy points using a custom scoring system:

- 10 rush yards = 1
- 10 receiving yards = 1
- 25 passing yards = 1
- Rush/Receiving TD = 6
- Passing TD = 4
- Interception = -2
- Completion = 0.3
- Incompletion = -0.3 (ATT - CMP)
- 10 kick or punt return yards = 1

Then rank Top-N candidate fantasy players at QB / RB / WR, with a matchup
context overlay (favourite spread and game total).

CLI examples:
    python cfb_fantasy_candidates.py
    python cfb_fantasy_candidates.py --leaders season --rows-per-pos 200 --topn 10
    python cfb_fantasy_candidates.py --start 2025-10-04 --days 6
    python cfb_fantasy_candidates.py --w-total 0.7 --w-spread 0.5
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple, Callable

import requests
from bs4 import BeautifulSoup, Tag
from loguru import logger


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _normalise_key(s: str) -> str:
    """
    Return a simple normalised key for fuzzy matching names/teams.

    Args:
        s (str): Input string.

    Returns:
        str: Lowercased, alnum-only string.
    """
    return "".join(ch.lower() for ch in (s or "") if ch.isalnum())


def _safe_float(x: Any) -> Optional[float]:
    """
    Convert anything to float safely, normalising half-points.

    Args:
        x (Any): Input value.

    Returns:
        Optional[float]: Parsed float or None.
    """
    if x is None:
        return None
    try:
        return float(str(x).strip().replace("½", ".5").replace(",", ""))
    except Exception:
        return None


def _parse_number(text: str) -> Optional[float]:
    """
    Parse a number from raw text that may include commas or extra chars.

    Args:
        text (str): Raw cell text.

    Returns:
        Optional[float]: Parsed float or None.
    """
    s = (text or "").strip().replace(",", "")
    m = re.search(r"-?\d+(\.\d+)?", s)
    return float(m.group()) if m else None


# -----------------------------------------------------------------------------
# Odds (Scoreboard JSON)
# -----------------------------------------------------------------------------

def _iso_to_compact(date: dt.date) -> str:
    """Convert a date to YYYYMMDD for ESPN's API."""
    return date.strftime("%Y%m%d")


def _build_dates_param(start: dt.date, end: dt.date) -> str:
    """Build "YYYYMMDD-YYYYMMDD" for the dates param."""
    return f"{_iso_to_compact(start)}-{_iso_to_compact(end)}"


def _get_scoreboard(
    start: dt.date, end: dt.date, groups: Iterable[int]
) -> Dict[str, Any]:
    """
    Query ESPN NCAAF scoreboard for the date range and groups.

    Args:
        start (datetime.date): Start date.
        end (datetime.date): End date.
        groups (Iterable[int]): ESPN group IDs (80≈FBS).

    Returns:
        Dict[str, Any]: Parsed JSON.
    """
    base = "https://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard"
    params = {"dates": _build_dates_param(start, end), "groups": ",".join(str(g) for g in groups) or "80"}
    logger.debug("Fetching scoreboard: {}", params)
    try:
        r = requests.get(base, params=params, timeout=25)
        r.raise_for_status()
        return r.json()
    except requests.HTTPError:
        # Retry per-group and merge events as a fallback.
        all_events: List[Dict[str, Any]] = []
        for g in groups:
            rg = requests.get(base, params={"dates": params["dates"], "groups": g}, timeout=25)
            rg.raise_for_status()
            all_events.extend((rg.json() or {}).get("events", []))
        return {"events": all_events}


def _extract_odds_from_event(event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Extract home/away names, favourite, spread (negative fav), and total.

    Args:
        event (Dict[str, Any]): ESPN event json.

    Returns:
        Optional[Dict[str, Any]]: Parsed odds row or None.
    """
    comp = (event.get("competitions") or [None])[0]
    if not comp:
        return None

    competitors = comp.get("competitors") or []
    if len(competitors) != 2:
        return None

    home = next((c for c in competitors if c.get("homeAway") == "home"), competitors[0])
    away = next((c for c in competitors if c.get("homeAway") == "away"), competitors[-1])

    def tname(c: Dict[str, Any]) -> str:
        t = c.get("team", {}) or {}
        return t.get("displayName") or t.get("shortDisplayName") or t.get("name") or "Unknown"

    home_team = tname(home)
    away_team = tname(away)
    kickoff = comp.get("date") or event.get("date")

    favourite_name: Optional[str] = None
    spread_value: Optional[float] = None
    total_value: Optional[float] = None
    odds_list = comp.get("odds") or event.get("odds") or []

    def infer_from_home_away_spreads(od: Dict[str, Any]) -> Tuple[Optional[str], Optional[float]]:
        h = od.get("homeTeamOdds") or {}
        a = od.get("awayTeamOdds") or {}
        h_sp = _safe_float(h.get("spread"))
        a_sp = _safe_float(a.get("spread"))
        if h_sp is not None and a_sp is not None:
            # more negative is the favourite
            return (home_team, h_sp) if h_sp <= a_sp else (away_team, a_sp)
        if h_sp is not None:
            return (home_team, h_sp) if h_sp < 0 else (away_team, -abs(h_sp))
        if a_sp is not None:
            return (away_team, a_sp) if a_sp < 0 else (home_team, -abs(a_sp))
        return None, None

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

        def norm(s: str) -> str:
            return "".join(ch.lower() for ch in s if ch.isalnum())

        mt, ht, at = norm(maybe_team), norm(home_team), norm(away_team)
        mapped = home_team if (mt in ht or ht in mt) else away_team if (mt in at or at in mt) else maybe_team
        return mapped, -abs(sp)

    for od in odds_list:
        if total_value is None:
            total_value = _safe_float(od.get("overUnder")) or total_value
        fav1, sp1 = infer_from_home_away_spreads(od)
        if fav1 and sp1 is not None:
            favourite_name, spread_value = fav1, sp1
        if (not favourite_name or spread_value is None) and od.get("details"):
            fav2, sp2 = infer_from_details(od)
            if fav2 and sp2 is not None:
                favourite_name, spread_value = fav2, sp2

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


def gather_odds(start: dt.date, end: dt.date, groups: Iterable[int]) -> List[Dict[str, Any]]:
    """
    Gather parsed odds rows for a date window and group(s).

    Args:
        start (datetime.date): Start date.
        end (datetime.date): End date.
        groups (Iterable[int]): ESPN group IDs.

    Returns:
        List[Dict[str, Any]]: One row per event (if odds exist).
    """
    data = _get_scoreboard(start, end, groups)
    rows: List[Dict[str, Any]] = []
    for ev in (data or {}).get("events", []):
        parsed = _extract_odds_from_event(ev)
        if parsed:
            rows.append(parsed)
    return rows


# -----------------------------------------------------------------------------
# Team mapping (abbr -> full display name)
# -----------------------------------------------------------------------------

TEAMS_API = "https://site.api.espn.com/apis/site/v2/sports/football/college-football/teams"

def fetch_team_display_map() -> Dict[str, str]:
    """
    Fetch an ESPN mapping from team abbreviation to full display name.

    Returns:
        Dict[str, str]: e.g., {"MRSH": "Marshall Thundering Herd", "TA&M": "Texas A&M Aggies"}.
    """
    r = requests.get(TEAMS_API, timeout=20)
    r.raise_for_status()
    data = r.json() or {}
    out: Dict[str, str] = {}
    leagues = ((data.get("sports") or [{}])[0].get("leagues") or [{}])
    teams = []
    for lg in leagues:
        teams.extend(lg.get("teams") or [])
    for t in teams:
        item = t.get("team") or {}
        abbr = item.get("abbreviation")
        disp = item.get("displayName") or item.get("shortDisplayName") or item.get("name")
        if abbr and disp:
            out[abbr] = disp
    return out


# -----------------------------------------------------------------------------
# Player leaders scraping (season-to-date) – SSR table routes
# -----------------------------------------------------------------------------

PASSING_URL = "https://www.espn.com/college-football/stats/player/_/view/offense/table/passing"
RUSHING_URL = "https://www.espn.com/college-football/stats/player/_/stat/rushing/table/rushing"
RECEIVING_URL = "https://www.espn.com/college-football/stats/player/_/stat/receiving/table/receiving"
RETURNING_URL = "https://www.espn.com/college-football/stats/player/_/view/special/stat/returning/table/returning"


@dataclass
class PlayerLine:
    """
    Container for per-player season totals relevant to fantasy scoring.

    Attributes:
        name (str): Player display name.
        team (str): ESPN team display name (mapped from abbreviation).
        pos (str): Position label when available (e.g., QB/RB/WR).
        cmp (float): Completions.
        att (float): Pass attempts.
        pass_yds (float): Passing yards.
        pass_td (float): Passing TDs.
        ints (float): Interceptions thrown.
        rush_yds (float): Rushing yards.
        rush_td (float): Rushing TDs.
        rec_yds (float): Receiving yards.
        rec_td (float): Receiving TDs.
        ret_yds (float): Total return yards (kick + punt).
    """
    name: str
    team: str
    pos: Optional[str] = None
    cmp: float = 0.0
    att: float = 0.0
    pass_yds: float = 0.0
    pass_td: float = 0.0
    ints: float = 0.0
    rush_yds: float = 0.0
    rush_td: float = 0.0
    rec_yds: float = 0.0
    rec_td: float = 0.0
    ret_yds: float = 0.0

    def key(self) -> Tuple[str, str]:
        """
        Return a normalised identity key for joins.

        Returns:
            Tuple[str, str]: (name_key, team_key).
        """
        return _normalise_key(self.name), _normalise_key(self.team)


def _fetch_html(url: str, params: Optional[Dict[str, Any]] = None) -> BeautifulSoup:
    """
    GET a page and return BeautifulSoup with browser-like headers.

    Args:
        url (str): Target URL.
        params (Optional[Dict[str, Any]]): Optional query parameters.

    Returns:
        BeautifulSoup: Parsed HTML document.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/129.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.espn.com/",
    }
    logger.debug("GET {} params={}", url, params)
    r = requests.get(url, params=params, headers=headers, timeout=25)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")


def _find_stats_table(soup: BeautifulSoup, required_cols: Iterable[str]) -> Optional[Tag]:
    """
    Find the first <table> whose header row contains all required columns.

    Args:
        soup (BeautifulSoup): Parsed document.
        required_cols (Iterable[str]): Column labels to find (e.g., {"NAME","CMP","ATT"}).

    Returns:
        Optional[Tag]: The matching table element or None.
    """
    required = {c.upper() for c in required_cols}
    for tbl in soup.find_all("table"):
        rows = tbl.find_all("tr")
        if not rows:
            continue
        hdr_cells = [c.get_text(" ", strip=True).upper() for c in rows[0].find_all(["th", "td"])]
        if required.issubset(set(hdr_cells)):
            return tbl
    return None


def _name_team_from_cell(td: Tag) -> Tuple[str, Optional[str]]:
    """
    Parse player name and team abbreviation from the 'NAME' cell.

    ESPN often places the team abbreviation in the same cell as the player name,
    typically as trailing text. We'll extract:
      - player name from the anchor text
      - team abbrev as the last ALL-CAPS token allowing &, - (e.g., TA&M, M-OH)

    Args:
        td (Tag): The 'NAME' cell (<td>).

    Returns:
        Tuple[str, Optional[str]]: (player_name, team_abbreviation or None)
    """
    a = td.find("a")
    name = a.get_text(" ", strip=True) if a else td.get_text(" ", strip=True)

    # Remove the name text from the full cell, then look for a plausible team token
    raw = td.get_text(" ", strip=True)
    tail = raw.replace(name, " ").strip()

    # Regex for tokens like ALA, TA&M, M-OH, BYU, UCF, etc.
    team_abbr = None
    if tail:
        candidates = re.findall(r"[A-Z]{2,4}(?:[-&][A-Z]{1,3})?", tail)
        if candidates:
            team_abbr = candidates[-1]

    return name, team_abbr


def _scrape_table_generic(
    url: str,
    required_cols: List[str],
    season: Optional[int],
    max_rows: int,
    parse_row: Callable[[Dict[str, int], Tag, Dict[str, str]], Optional[PlayerLine]],
) -> Dict[Tuple[str, str], PlayerLine]:
    """
    Generic helper to fetch a stats page, find the right table, and parse rows.

    Args:
        url (str): ESPN stats URL.
        required_cols (List[str]): Header labels required to identify the table.
        season (Optional[int]): Season year (ESPN defaults to current if None).
        max_rows (int): Cap on rows to parse.
        parse_row (callable): (header_map, tr, team_map) -> PlayerLine|None.

    Returns:
        Dict[Tuple[str, str], PlayerLine]: Mapping keyed by (name_key, team_key).
    """
    params = {}
    if season:
        params["season"] = season

    soup = _fetch_html(url, params=params)
    table = _find_stats_table(soup, required_cols)
    if not table:
        logger.warning("No stats table found at {}", url)
        return {}

    header_cells = [c.get_text(" ", strip=True).upper() for c in table.find_all("tr")[0].find_all(["th", "td"])]
    header_map = {label: idx for idx, label in enumerate(header_cells)}

    team_map = fetch_team_display_map()

    out: Dict[Tuple[str, str], PlayerLine] = {}
    for tr in table.find_all("tr")[1 : max_rows + 1]:
        pl = parse_row(header_map, tr, team_map)
        if pl:
            out[pl.key()] = pl
    logger.info("Parsed {} rows from {}", len(out), url)
    return out


def _td_text(tds: List[Tag], idx: Optional[int]) -> str:
    """Safely get text from a td by index ('' if unavailable)."""
    if idx is None or idx < 0 or idx >= len(tds):
        return ""
    return tds[idx].get_text(" ", strip=True)


def _val_from_header(tds: List[Tag], header_map: Dict[str, int], label: str) -> float:
    """Float value from a header label (0.0 if missing)."""
    return _parse_number(_td_text(tds, header_map.get(label))) or 0.0


def scrape_passing_leaders(season: Optional[int] = None, max_rows: int = 400) -> Dict[Tuple[str, str], PlayerLine]:
    """
    Scrape passing leaders with CMP, ATT, YDS, TD, INT.
    """
    def parse_row(header_map: Dict[str, int], tr: Tag, team_map: Dict[str, str]) -> Optional[PlayerLine]:
        tds = tr.find_all("td")
        if not tds:
            return None

        # Identify the NAME (or PLAYER) column dynamically
        name_idx = header_map.get("NAME", header_map.get("PLAYER"))
        if name_idx is None or name_idx >= len(tds):
            return None

        name_td = tds[name_idx]
        name, team_abbr = _name_team_from_cell(name_td)
        team_display = team_map.get(team_abbr or "", team_abbr or "Unknown")

        pl = PlayerLine(
            name=name,
            team=team_display,
            pos="QB",
            cmp=_val_from_header(tds, header_map, "CMP"),
            att=_val_from_header(tds, header_map, "ATT"),
            pass_yds=_val_from_header(tds, header_map, "YDS"),
            pass_td=_val_from_header(tds, header_map, "TD"),
            ints=_val_from_header(tds, header_map, "INT"),
        )
        return pl

    required = ["NAME", "CMP", "ATT", "YDS", "TD", "INT"]
    return _scrape_table_generic(PASSING_URL, required, season, max_rows, parse_row)


def scrape_rushing_leaders(season: Optional[int] = None, max_rows: int = 800) -> Dict[Tuple[str, str], PlayerLine]:
    """
    Scrape rushing leaders (YDS, TD).
    """
    def parse_row(header_map: Dict[str, int], tr: Tag, team_map: Dict[str, str]) -> Optional[PlayerLine]:
        tds = tr.find_all("td")
        if not tds:
            return None

        name_idx = header_map.get("NAME", header_map.get("PLAYER"))
        if name_idx is None or name_idx >= len(tds):
            return None

        name_td = tds[name_idx]
        name, team_abbr = _name_team_from_cell(name_td)
        team_display = team_map.get(team_abbr or "", team_abbr or "Unknown")

        pl = PlayerLine(
            name=name,
            team=team_display,
            pos="RB",  # default; later bucket by dominance if needed
            rush_yds=_val_from_header(tds, header_map, "YDS"),
            rush_td=_val_from_header(tds, header_map, "TD"),
        )
        return pl

    required = ["NAME", "ATT", "YDS", "TD"]
    return _scrape_table_generic(RUSHING_URL, required, season, max_rows, parse_row)


def scrape_receiving_leaders(season: Optional[int] = None, max_rows: int = 800) -> Dict[Tuple[str, str], PlayerLine]:
    """
    Scrape receiving leaders (YDS, TD).
    """
    def parse_row(header_map: Dict[str, int], tr: Tag, team_map: Dict[str, str]) -> Optional[PlayerLine]:
        tds = tr.find_all("td")
        if not tds:
            return None

        name_idx = header_map.get("NAME", header_map.get("PLAYER"))
        if name_idx is None or name_idx >= len(tds):
            return None

        name_td = tds[name_idx]
        name, team_abbr = _name_team_from_cell(name_td)
        team_display = team_map.get(team_abbr or "", team_abbr or "Unknown")

        pl = PlayerLine(
            name=name,
            team=team_display,
            pos="WR",
            rec_yds=_val_from_header(tds, header_map, "YDS"),
            rec_td=_val_from_header(tds, header_map, "TD"),
        )
        return pl

    required = ["NAME", "REC", "YDS", "TD"]
    return _scrape_table_generic(RECEIVING_URL, required, season, max_rows, parse_row)


def scrape_returning_leaders(season: Optional[int] = None, max_rows: int = 800) -> Dict[Tuple[str, str], PlayerLine]:
    """
    Scrape special teams 'Returning' leaders (kickoffs and punts). We sum kick and punt
    return yards as a single 'ret_yds' (fantasy scoring uses yards only).
    """
    def parse_row(header_map: Dict[str, int], tr: Tag, team_map: Dict[str, str]) -> Optional[PlayerLine]:
        tds = tr.find_all("td")
        if not tds:
            return None

        name_idx = header_map.get("NAME", header_map.get("PLAYER"))
        if name_idx is None or name_idx >= len(tds):
            return None

        name_td = tds[name_idx]
        name, team_abbr = _name_team_from_cell(name_td)
        team_display = team_map.get(team_abbr or "", team_abbr or "Unknown")

        # First two YDS columns usually correspond to Kick and Punt return yards
        hdr_items = list(header_map.items())
        yds_cols = [idx for lbl, idx in hdr_items if lbl == "YDS"]
        def td_val(i: Optional[int]) -> float:
            if i is None or i >= len(tds):
                return 0.0
            return _parse_number(tds[i].get_text(" ", strip=True)) or 0.0

        kick_y = td_val(yds_cols[0] if len(yds_cols) > 0 else None)
        punt_y = td_val(yds_cols[1] if len(yds_cols) > 1 else None)

        pl = PlayerLine(
            name=name,
            team=team_display,
            ret_yds=kick_y + punt_y,
        )
        return pl

    required = ["NAME", "YDS"]
    return _scrape_table_generic(RETURNING_URL, required, season, max_rows, parse_row)


def merge_leaders(
    passing: Dict[Tuple[str, str], PlayerLine],
    rushing: Dict[Tuple[str, str], PlayerLine],
    receiving: Dict[Tuple[str, str], PlayerLine],
    returning: Dict[Tuple[str, str], PlayerLine],
) -> Dict[Tuple[str, str], PlayerLine]:
    """
    Merge the four leader dicts into a single PlayerLine per (name, team).
    """
    merged: Dict[Tuple[str, str], PlayerLine] = {}
    keys = set(passing) | set(rushing) | set(receiving) | set(returning)
    for k in keys:
        base = passing.get(k) or rushing.get(k) or receiving.get(k) or returning.get(k)
        assert base is not None
        pl = PlayerLine(name=base.name, team=base.team, pos=base.pos)
        if k in passing:
            p = passing[k]
            pl.cmp, pl.att, pl.pass_yds, pl.pass_td, pl.ints = p.cmp, p.att, p.pass_yds, p.pass_td, p.ints
            pl.pos = pl.pos or "QB"
        if k in rushing:
            ru = rushing[k]
            pl.rush_yds += ru.rush_yds
            pl.rush_td += ru.rush_td
            if pl.pos is None:
                pl.pos = "RB"
        if k in receiving:
            rc = receiving[k]
            pl.rec_yds += rc.rec_yds
            pl.rec_td += rc.rec_td
            if pl.pos is None:
                pl.pos = "WR"
        if k in returning:
            rt = returning[k]
            pl.ret_yds += rt.ret_yds
        merged[k] = pl
    return merged


# -----------------------------------------------------------------------------
# Fantasy scoring + context ranking
# -----------------------------------------------------------------------------

def fantasy_points(pl: PlayerLine) -> float:
    """
    Compute fantasy points per the provided scoring framework.
    """
    rush_pts = (pl.rush_yds / 10.0) + (pl.rush_td * 6.0)
    rec_pts = (pl.rec_yds / 10.0) + (pl.rec_td * 6.0)
    pass_pts = (pl.pass_yds / 25.0) + (pl.pass_td * 4.0) + (pl.ints * -2.0) + (pl.cmp * 0.3) + ((pl.att - pl.cmp) * -0.3)
    ret_pts = (pl.ret_yds / 10.0)
    return rush_pts + rec_pts + pass_pts + ret_pts


@dataclass
class GameContext:
    """
    Betting context for a specific team.
    """
    team: str
    opponent: str
    spread: Optional[float]
    total: Optional[float]
    is_favourite: Optional[bool]


def build_team_context(odds_rows: List[Dict[str, Any]]) -> Dict[str, GameContext]:
    """
    Build a mapping from normalised team name -> GameContext.
    """
    ctx: Dict[str, GameContext] = {}
    for r in odds_rows:
        h, a = r["home_team"], r["away_team"]
        fav, sp, tot = r.get("favourite"), r.get("spread"), r.get("total")
        ctx[_normalise_key(h)] = GameContext(
            team=h,
            opponent=a,
            spread=sp if fav == h else (-sp if (sp is not None and fav == a) else None),
            total=tot,
            is_favourite=True if fav == h else False if fav == a else None,
        )
        ctx[_normalise_key(a)] = GameContext(
            team=a,
            opponent=h,
            spread=sp if fav == a else (-sp if (sp is not None and fav == h) else None),
            total=tot,
            is_favourite=True if fav == a else False if fav == h else None,
        )
    return ctx


def _z(values: List[float], x: float) -> float:
    """
    Compute a z-score with guards.
    """
    if not values:
        return 0.0
    mu = sum(values) / len(values)
    var = sum((v - mu) ** 2 for v in values) / max(1, len(values) - 1)
    sd = math.sqrt(var)
    if sd < 1e-8:
        return 0.0
    return (x - mu) / sd


@dataclass
class Ranked:
    """
    Ranked result container.
    """
    player: PlayerLine
    base_fp: float
    score: float
    ctx: Optional[GameContext] = None
    extras: Dict[str, Any] = field(default_factory=dict)


def rank_with_context(
    players: List[PlayerLine],
    team_ctx: Dict[str, GameContext],
    w_spread: float = 0.5,
    w_total: float = 0.7,
    topn: int = 10,
) -> List[Ranked]:
    """
    Rank players by fantasy points plus matchup context.
    """
    base_vals = [fantasy_points(p) for p in players]
    totals = [c.total for c in team_ctx.values() if isinstance(c.total, (int, float))]
    spreads = [c.spread for c in team_ctx.values() if isinstance(c.spread, (int, float))]
    inv_spreads = [(-s) for s in spreads]  # more negative fav => higher value

    out: List[Ranked] = []
    for p in players:
        fp = fantasy_points(p)
        ctx = team_ctx.get(_normalise_key(p.team))
        spread_term = 0.0
        total_term = 0.0
        if ctx and isinstance(ctx.spread, (int, float)):
            spread_term = _z(inv_spreads, -ctx.spread)
        if ctx and isinstance(ctx.total, (int, float)):
            total_term = _z(totals, ctx.total)
        score = _z(base_vals, fp) + w_spread * spread_term + w_total * total_term
        out.append(Ranked(player=p, base_fp=fp, score=score, ctx=ctx))
    out.sort(key=lambda r: r.score, reverse=True)
    return out[:topn]


# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------

def main() -> int:
    """
    CLI entrypoint.
    """
    parser = argparse.ArgumentParser(description="CFB fantasy candidates using custom scoring + betting context.")
    today = dt.date.today()
    parser.add_argument("--start", type=str, default=today.isoformat(), help="Odds window start date (YYYY-MM-DD).")
    parser.add_argument("--end", type=str, default=(today + dt.timedelta(days=6)).isoformat(),
                        help="Odds window end date (YYYY-MM-DD).")
    parser.add_argument("--days", type=int, default=None, help="Shortcut: days after start (overrides --end).")
    parser.add_argument("--groups", type=int, nargs="+", default=[80], help="ESPN groups (80≈FBS).")
    parser.add_argument("--season", type=int, default=None, help="Season year (leaders pages).")
    parser.add_argument("--rows-per-pos", type=int, default=400, help="Rows to parse per table (cap for speed).")
    parser.add_argument("--topn", type=int, default=10, help="Top-N per position to output.")
    parser.add_argument("--w-spread", type=float, default=0.5, help="Weight for spread term.")
    parser.add_argument("--w-total", type=float, default=0.7, help="Weight for total term.")
    args = parser.parse_args()

    start_date = dt.date.fromisoformat(args.start)
    end_date = (start_date + dt.timedelta(days=args.days)) if args.days is not None else dt.date.fromisoformat(args.end)

    logger.info("Building odds window: {} to {} (groups={})", start_date, end_date, args.groups)
    odds_rows = gather_odds(start_date, end_date, args.groups)
    if not odds_rows:
        logger.error("No odds found in window.")
        return 2
    team_ctx = build_team_context(odds_rows)

    logger.info("Scraping season leaders (passing/rushing/receiving/returning)...")
    passing = scrape_passing_leaders(season=args.season, max_rows=args.rows_per_pos)
    rushing = scrape_rushing_leaders(season=args.season, max_rows=args.rows_per_pos)
    receiving = scrape_receiving_leaders(season=args.season, max_rows=args.rows_per_pos)
    returning = scrape_returning_leaders(season=args.season, max_rows=args.rows_per_pos)

    merged = merge_leaders(passing, rushing, receiving, returning)

    # Partition roughly by position label; keep guardrails:
    qbs: List[PlayerLine] = []
    rbs: List[PlayerLine] = []
    wrs: List[PlayerLine] = []
    for pl in merged.values():
        if pl.pos == "QB":
            qbs.append(pl)
        elif pl.pos == "RB":
            rbs.append(pl)
        elif pl.pos == "WR":
            wrs.append(pl)
        else:
            if pl.pass_yds > max(pl.rush_yds, pl.rec_yds):
                qbs.append(pl)
            elif pl.rush_yds >= pl.rec_yds:
                rbs.append(pl)
            else:
                wrs.append(pl)

    print("\n=== Odds Window ===")
    print(f"{start_date.isoformat()} to {end_date.isoformat()} | groups={args.groups}")
    print(f"Games with spreads: {sum(1 for r in odds_rows if isinstance(r.get('spread'), (int, float)))} | "
          f"with totals: {sum(1 for r in odds_rows if isinstance(r.get('total'), (int, float)))}")

    # Rank per position
    for label, bucket in [("QB", qbs), ("RB", rbs), ("WR", wrs)]:
        ranked = rank_with_context(bucket, team_ctx, w_spread=args.w_spread, w_total=args.w_total, topn=args.topn)
        print(f"\n=== Top {args.topn} {label} (custom fantasy scoring + context) ===")
        for i, r in enumerate(ranked, start=1):
            p, fp, sc, ctx = r.player, r.base_fp, r.score, r.ctx
            ctx_str = "no-line"
            if ctx:
                fav_tag = "fav" if ctx.is_favourite else "dog" if ctx.is_favourite is not None else "n/a"
                ctx_str = f"opp={ctx.opponent}, spread={ctx.spread if ctx.spread is not None else 'NA'}, " \
                          f"total={ctx.total if ctx.total is not None else 'NA'}, {fav_tag}"
            print(
                f"{i:>2}. {p.name} ({p.team}) | FP={fp:.1f} | score={sc:.2f} "
                f"| pass: {p.cmp:.0f}/{p.att:.0f} {p.pass_yds:.0f}y {p.pass_td:.0f}TD {p.ints:.0f}INT "
                f"| rush: {p.rush_yds:.0f}y {p.rush_td:.0f}TD | rec: {p.rec_yds:.0f}y {p.rec_td:.0f}TD "
                f"| ret: {p.ret_yds:.0f}y | {ctx_str}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
