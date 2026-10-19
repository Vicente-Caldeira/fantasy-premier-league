"""Read endpoints over the store, plus squad evaluation and transfer planning."""

from collections.abc import Callable
from typing import Annotated, Any

import polars as pl
from fastapi import APIRouter, Depends, HTTPException

from fpl_api import schemas
from fpl_api.store import Snapshot, Store
from fpl_core.fpl_api import FplApiClient
from fpl_models import optimizer, team_import

PLAN_TIME_LIMIT_SECONDS = 20


def _m(tenths: float | None) -> float:
    return round((tenths or 0) / 10, 1)


def _f(value: Any, digits: int = 3) -> float | None:  # noqa: ANN401 - polars scalars
    return None if value is None else round(float(value), digits)


def build_router(get_store: Callable[[], Store]) -> APIRouter:  # noqa: PLR0915 - one router
    router = APIRouter(prefix="/api")

    def snapshot(store: Annotated[Store, Depends(get_store)]) -> Snapshot:
        return store.get()

    Snap = Annotated[Snapshot, Depends(snapshot)]  # noqa: N806 - type alias

    def opponents(snap: Snapshot) -> dict[tuple[str, int], list[schemas.Opponent]]:
        out: dict[tuple[str, int], list[schemas.Opponent]] = {}
        for gw, home, away in snap.fixtures.select(
            "gameweek", "home_team_key", "away_team_key"
        ).iter_rows():
            out.setdefault((home, gw), []).append(schemas.Opponent(team_key=away, is_home=True))
            out.setdefault((away, gw), []).append(schemas.Opponent(team_key=home, is_home=False))
        return out

    def players(snap: Snapshot, frame: pl.DataFrame | None = None) -> list[schemas.Player]:
        horizon = snap.horizon
        expected = snap.expected.select("player_code", "gameweek", "expected_points")
        wide = expected.pivot(on="gameweek", index="player_code", values="expected_points")
        nxt = snap.expected.filter(pl.col("gameweek") == horizon[0]).select(
            "player_code", "p_haul", "sim_p10", "sim_p90", "p_60"
        )
        rows = (
            (frame if frame is not None else snap.players)
            .join(wide, on="player_code", how="left")
            .join(nxt, on="player_code", how="left")
        )
        out = []
        for r in rows.iter_rows(named=True):
            xp = [round(r.get(str(g)) or 0.0, 2) for g in horizon]
            out.append(
                schemas.Player(
                    code=r["player_code"],
                    element=r["element"],
                    name=r["web_name"],
                    position=r["position"],
                    team_key=r["team_key"],
                    price=_m(r["price"]),
                    ownership=float(r["selected_by_percent"] or 0),
                    status=r["status"],
                    news=r["news"] or "",
                    chance_of_playing=r["chance_of_playing_next_round"],
                    xp=xp,
                    xp_total=round(sum(xp), 2),
                    p_haul=_f(r["p_haul"]),
                    p10=_f(r["sim_p10"], 1),
                    p90=_f(r["sim_p90"], 1),
                    p_60=_f(r["p_60"]),
                    p_rise=_f(r["p_rise"]),
                    p_fall=_f(r["p_fall"]),
                    net_transfers=r["net_transfers"],
                    season=schemas.SeasonStats(
                        points=r["points"] or 0,
                        minutes=r["minutes"] or 0,
                        appearances=r["appearances"] or 0,
                        starts=r["starts"] or 0,
                        goals=r["goals"] or 0,
                        assists=r["assists"] or 0,
                        xg=round(r["xg"] or 0.0, 2),
                        xa=round(r["xa"] or 0.0, 2),
                        clean_sheets=r["clean_sheets"] or 0,
                        bonus=r["bonus"] or 0,
                        saves=r["saves"] or 0,
                        defensive_contribution=r["defensive_contribution"] or 0,
                        form=round(r["form"] or 0.0, 1),
                    ),
                )
            )
        return out

    def club_fixtures(snap: Snapshot, key: str) -> list[schemas.ClubFixture]:
        out = []
        for r in snap.fixtures.sort("kickoff_utc").iter_rows(named=True):
            if key not in (r["home_team_key"], r["away_team_key"]):
                continue
            home = r["home_team_key"] == key
            out.append(
                schemas.ClubFixture(
                    gameweek=r["gameweek"],
                    opponent=r["away_team_key"] if home else r["home_team_key"],
                    is_home=home,
                    goals_for=round(r["lambda_home"] if home else r["lambda_away"], 2),
                    goals_against=round(r["lambda_away"] if home else r["lambda_home"], 2),
                    clean_sheet=round(r["home_clean_sheet"] if home else r["away_clean_sheet"], 3),
                    win=round(r["home_win"] if home else r["away_win"], 3),
                    draw=round(r["draw"], 3),
                    loss=round(r["away_win"] if home else r["home_win"], 3),
                )
            )
        return out

    def record(snap: Snapshot, key: str) -> schemas.ClubRecord:
        won = drawn = lost = gf = ga = 0
        xgf = xga = 0.0
        for r in snap.results.iter_rows(named=True):
            if key not in (r["home_team_key"], r["away_team_key"]):
                continue
            home = r["home_team_key"] == key
            f, a = (
                (r["home_goals"], r["away_goals"]) if home else (r["away_goals"], r["home_goals"])
            )
            xf, xa = (r["home_xg"], r["away_xg"]) if home else (r["away_xg"], r["home_xg"])
            gf, ga, xgf, xga = gf + f, ga + a, xgf + (xf or 0), xga + (xa or 0)
            won, drawn, lost = won + (f > a), drawn + (f == a), lost + (f < a)
        return schemas.ClubRecord(
            played=won + drawn + lost, won=won, drawn=drawn, lost=lost, goals_for=gf,
            goals_against=ga, xg_for=round(xgf, 1), xg_against=round(xga, 1),
        )  # fmt: skip

    def club(snap: Snapshot, r: dict[str, Any]) -> schemas.Club:
        return schemas.Club(
            key=r["team_key"], name=r["name"], short_name=r["short_name"],
            fpl_team_code=r["fpl_team_code"], attack=round(r["attack"], 3),
            defence=round(r["defence"], 3), record=record(snap, r["team_key"]),
            fixtures=club_fixtures(snap, r["team_key"]),
        )  # fmt: skip

    def fixture(r: dict[str, Any]) -> schemas.Fixture:
        fields = schemas.Fixture.model_fields
        values = {
            k: v for k, v in r.items() if k in fields and k != "kickoff_utc" and v is not None
        }
        return schemas.Fixture(
            **values, home=r["home_team_key"], away=r["away_team_key"],
            kickoff_utc=r["kickoff_utc"].isoformat(),
        )  # fmt: skip

    @router.get("/meta", tags=["data"])
    def meta(snap: Snap) -> schemas.Meta:
        gws = snap.gameweeks
        current = gws.filter(pl.col("is_current"))
        nxt = gws.filter(pl.col("is_next"))
        return schemas.Meta(
            season=snap.season,
            current_gameweek=current["gameweek"].item() if current.height else None,
            next_gameweek=nxt["gameweek"].item() if nxt.height else None,
            next_deadline=nxt["deadline_utc"].item().isoformat() if nxt.height else None,
            horizon=snap.horizon,
            predictions_as_of=snap.predictions_as_of,
        )

    @router.get("/players", tags=["data"])
    def list_players(snap: Snap) -> list[schemas.Player]:
        return players(snap)

    @router.get("/players/{code}", tags=["data"])
    def player_detail(code: int, snap: Snap) -> schemas.PlayerDetail:
        row = snap.players.filter(pl.col("player_code") == code)
        if row.height == 0:
            raise HTTPException(404, f"no player {code} in this season's squads")
        team = row["team_key"].item()
        opp = opponents(snap)
        upcoming = [
            schemas.UpcomingGameweek(
                gameweek=r["gameweek"],
                opponents=opp.get((team, r["gameweek"]), []),
                xp=round(r["expected_points"], 2),
                p_play=_f(r["p_play"]),
                p_60=_f(r["p_60"]),
                e_goals=_f(r["e_goals_scored"]),
                e_assists=_f(r["e_assists"]),
                p_clean_sheet=_f(r["p_clean_sheet"]),
                availability=_f(r["availability"]),
                p10=_f(r["sim_p10"], 1),
                p90=_f(r["sim_p90"], 1),
                p_haul=_f(r["p_haul"]),
            )
            for r in snap.expected.filter(pl.col("player_code") == code)
            .sort("gameweek")
            .iter_rows(named=True)
        ]
        matches = [
            schemas.MatchRow(
                season=r["season"],
                gameweek=r["gameweek"],
                opponent=r["opponent_team_key"],
                is_home=r["is_home"],
                minutes=r["minutes"],
                points=r["total_points"],
                goals=r["goals_scored"],
                assists=r["assists"],
                xg=_f(r["expected_goals"], 2),
                xa=_f(r["expected_assists"], 2),
                bonus=r["bonus"],
                bps=r["bps"],
                clean_sheet=r["clean_sheets"],
                price=_m(r["price"]),
                selected=r["selected"],
            )
            for r in snap.history.filter(pl.col("player_code") == code)
            .sort("kickoff_utc")
            .iter_rows(named=True)
        ]
        past = [
            schemas.PastSeason(**{**r, "max_price": _m(r["max_price"])})
            for r in snap.past_seasons.filter(pl.col("player_code") == code)
            .sort("season")
            .drop("player_code")
            .iter_rows(named=True)
        ]
        return schemas.PlayerDetail(
            player=players(snap, row)[0], upcoming=upcoming, matches=matches, past_seasons=past
        )

    @router.get("/fixtures", tags=["data"])
    def fixtures(snap: Snap) -> list[schemas.Fixture]:
        played = [fixture(r) for r in snap.results.sort("kickoff_utc").iter_rows(named=True)]
        upcoming = [fixture(r) for r in snap.fixtures.sort("kickoff_utc").iter_rows(named=True)]
        return played + upcoming

    @router.get("/clubs", tags=["data"])
    def clubs(snap: Snap) -> list[schemas.Club]:
        return [club(snap, r) for r in snap.clubs.sort("name").iter_rows(named=True)]

    @router.get("/clubs/{key}", tags=["data"])
    def club_detail(key: str, snap: Snap) -> schemas.ClubDetail:
        row = snap.clubs.filter(pl.col("team_key") == key)
        if row.height == 0:
            raise HTTPException(404, f"no club {key}")
        involved = (pl.col("home_team_key") == key) | (pl.col("away_team_key") == key)
        return schemas.ClubDetail(
            club=club(snap, row.row(0, named=True)),
            results=[
                fixture(r)
                for r in snap.results.filter(involved).sort("kickoff_utc").iter_rows(named=True)
            ],
            players=players(snap, snap.players.filter(pl.col("team_key") == key)),
        )  # fmt: skip

    @router.get("/entry/{entry_id}", tags=["team"])
    def entry(entry_id: int, snap: Snap) -> schemas.Entry:
        nxt = snap.gameweeks.filter(pl.col("is_next"))
        if nxt.height == 0:
            raise HTTPException(409, "no upcoming gameweek")
        next_gw = int(nxt["gameweek"].item())
        try:
            with FplApiClient() as client:
                summary = client.get_json(f"entry/{entry_id}/")
                data = team_import.EntryData(
                    history=client.entry_history(entry_id),
                    transfers=client.entry_transfers(entry_id),
                    picks=client.entry_picks(entry_id, next_gw - 1),
                )
        except Exception as error:
            raise HTTPException(502, f"FPL API: {error}") from error
        squads = snap.players
        gw1 = snap.history.filter(
            (pl.col("season") == snap.season) & (pl.col("gameweek") == 1)
        ).select("element", "price")
        lookup = team_import.PlayerLookup(
            element_to_code=dict(zip(squads["element"], squads["player_code"], strict=True)),
            current_prices=dict(zip(squads["element"], squads["price"], strict=True)),
            gameweek_one_prices=dict(gw1.unique("element").iter_rows()),
        )
        team = team_import.import_team(entry_id, data, lookup, next_gw)
        picks = sorted(data.picks["picks"], key=lambda p: p["position"])
        code = lookup.element_to_code
        return schemas.Entry(
            id=entry_id,
            name=summary.get("name", ""),
            manager=" ".join(
                summary.get(k, "") for k in ("player_first_name", "player_last_name")
            ).strip(),
            overall_rank=summary.get("summary_overall_rank"),
            total_points=summary.get("summary_overall_points", 0),
            squad=[code[p["element"]] for p in picks],
            lineup=[code[p["element"]] for p in picks if p["position"] <= 11],
            captain=next((code[p["element"]] for p in picks if p["is_captain"]), None),
            vice_captain=next((code[p["element"]] for p in picks if p["is_vice_captain"]), None),
            bank=_m(team.bank),
            free_transfers=team.free_transfers,
            selling_prices={c: _m(v) for c, v in team.selling_prices.items()},
            next_gameweek=next_gw,
            chips_used=[f"{c['name']} (GW{c['event']})" for c in data.history.get("chips", [])],
            history=[
                schemas.GameweekHistory(
                    gameweek=w["event"],
                    points=w["points"],
                    total_points=w["total_points"],
                    overall_rank=w.get("overall_rank"),
                    transfers=w["event_transfers"],
                    hits_cost=w["event_transfers_cost"],
                    bank=_m(w["bank"]),
                    value=_m(w["value"]),
                )
                for w in data.history.get("current", [])
            ],
        )

    def solve(snap: Snapshot, request: schemas.SquadRequest, **kwargs: Any) -> schemas.Plan:  # noqa: ANN401
        gameweeks = snap.horizon[: request.horizon]
        unknown = set(request.squad) - set(snap.players["player_code"])
        if unknown:
            raise HTTPException(422, f"unknown players: {sorted(unknown)}")
        try:
            plan = optimizer.optimise(
                snap.players.select("player_code", "position", "team_key", "price"),
                snap.expected.filter(pl.col("gameweek").is_in(gameweeks)).select(
                    "player_code", "gameweek", "expected_points"
                ),
                current_squad=request.squad,
                settings=optimizer.Settings(time_limit_seconds=PLAN_TIME_LIMIT_SECONDS),
                **kwargs,
            )
        except optimizer.InfeasibleError as error:
            raise HTTPException(422, str(error)) from error
        return schemas.Plan(
            status=plan.status,
            gameweeks=[
                schemas.PlannedGameweek(
                    gameweek=g.gameweek,
                    squad=g.squad,
                    lineup=g.lineup,
                    captain=g.captain,
                    vice_captain=g.vice_captain,
                    transfers_in=g.transfers_in,
                    transfers_out=g.transfers_out,
                    free_transfers=g.free_transfers,
                    hits=g.hits,
                    xp=round(g.expected_points, 2),
                )
                for g in plan.gameweeks
            ],
            total_xp=round(plan.total_expected_points, 2),
        )

    @router.post("/squad/evaluate", tags=["team"])
    def evaluate(request: schemas.SquadRequest, snap: Snap) -> schemas.Plan:
        """Best lineup and captain each gameweek for a fixed squad (no transfers)."""
        if len(set(request.squad)) != len(request.squad):
            raise HTTPException(422, "duplicate players in squad")
        return solve(snap, request, keep=request.squad, free_transfers=0, bank=10_000)

    @router.post("/plan", tags=["team"])
    def plan(request: schemas.PlanRequest, snap: Snap) -> schemas.Plan:
        """Optimal transfers, lineups and captains over the horizon from a squad."""
        return solve(
            snap, request,
            bank=round(request.bank * 10), free_transfers=request.free_transfers,
            selling_prices={c: round(p * 10) for c, p in request.selling_prices.items()},
            keep=request.keep, ban=request.ban, wildcard=request.wildcard,
        )  # fmt: skip

    return router
