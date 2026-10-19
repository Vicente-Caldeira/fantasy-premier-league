"""Season simulation: manage an FPL team through a past season, decisions made blind.

At every deadline g, using only information available before it:
    1. the player pool: everyone registered this season so far, with his latest price,
       club and position as of g (players in a blank gameweek stay in the pool)
    2. expected points for g with the next-gameweek model (crowd signal included: that
       window is complete at the deadline), and for g+1..g+H-1 with the planning model
       from each player's latest state as of g, with team strength from ratings
       fitted on matches before g (no odds exist that far ahead)
    3. the optimiser chooses transfers, XI and captain from the current squad, bank and
       free transfers
    4. the real gameweek is scored: captain doubled (vice-captain if the captain didn't
       play), automatic substitutions in bench order keeping a valid formation, hits
       deducted; squad, bank (real price moves, FPL selling rule) and free transfers
       roll on

The first deadline builds the squad from scratch. Chips aren't used.
"""

from collections import Counter
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime
from functools import partial

import numpy as np
import polars as pl

from fpl_models import optimizer, player_model, simulator, team_strength
from fpl_models.player_features import CROWD_ALL
from fpl_models.team_import import selling_price

# Context that belongs to the fixture, not to the player's state.
FIXTURE_CONTEXT = (
    "is_home", "n_fixtures", "team_lambda", "opponent_lambda", "p_clean_sheet",
    "team_rest_days", "opponent_rest_days", "team_matches_prev_14d", "opponent_matches_prev_14d",
)  # fmt: skip


@dataclass
class SeasonResult:
    season: str
    points: list[int] = field(default_factory=list)  # per gameweek, after hits
    hits: list[int] = field(default_factory=list)

    @property
    def total(self) -> int:
        return sum(self.points)


def auto_subs(
    lineup: list[int], bench: list[int], minutes: dict[int, int], position: dict[int, str]
) -> list[int]:
    """Non-playing starters are replaced in bench order by bench players who played,
    keeping a valid formation; goalkeeper only for goalkeeper."""
    team = list(lineup)
    for starter in [p for p in lineup if minutes.get(p, 0) == 0]:
        for sub in bench:
            if sub in team or minutes.get(sub, 0) == 0:
                continue
            if (position[starter] == "GKP") != (position[sub] == "GKP"):
                continue
            candidate = [sub if p == starter else p for p in team]
            counts = Counter(position[p] for p in candidate)
            if all(counts[pos] >= minimum for pos, minimum in optimizer.LINEUP_MIN.items()):
                team = candidate
                break
    return team


def score(
    gw: optimizer.Gameweek,
    bench: list[int],
    actual: dict[int, int],
    minutes: dict[int, int],
    position: dict[int, str],
) -> int:
    team = auto_subs(gw.lineup, bench, minutes, position)
    captain = gw.captain if minutes.get(gw.captain, 0) > 0 else gw.vice_captain
    return sum(actual.get(p, 0) for p in team) + (actual.get(captain, 0) if captain in team else 0)


@dataclass(frozen=True)
class Week:
    """Everything a path needs at one deadline. It doesn't depend on the path's squad,
    so it is computed once per config and season and shared by every path."""

    gameweek: int
    pool: pl.DataFrame  # player_code, position, team_key, price
    xp: pl.DataFrame  # player_code, gameweek, expected_points (this + later gameweeks)
    upside: pl.DataFrame  # player_code, sim_p90, p_haul (this gameweek)
    actual: dict[int, int]
    minutes: dict[int, int]


@dataclass
class PathRecord:
    """One gameweek of one path, enough to rescore other captain rules (E2)."""

    gameweek: int
    squad: list[int]
    lineup: list[int]
    bench: list[int]
    captain: int
    vice_captain: int
    hits: int
    points: int  # after hits
    chip: str | None = None


# Chips can be played once in each half of the season (2025-26 rules).
FIRST_HALF_END = 19


@dataclass
class SeasonSimulator:
    features: pl.DataFrame  # player_features.build_features (all fixture rows)
    gameweeks: pl.DataFrame  # data.player_gameweeks (truth: points, minutes)
    matches: pl.DataFrame  # team_strength.load_matches
    config: player_model.ModelConfig
    horizon: int = 3
    refit_every: int = 4
    settings: optimizer.Settings = field(
        default_factory=lambda: optimizer.Settings(time_limit_seconds=30)
    )

    def run(self, season: str) -> SeasonResult:
        return play(season, self.prepare(season), self.settings)[0]

    def prepare(self, season: str) -> list[Week]:
        """Pool, expected points, upside and truth per deadline, decided blind."""
        deadlines = (
            self.gameweeks.filter(pl.col("season") == season)
            .select("gameweek", "deadline_utc")
            .unique()
            .sort("deadline_utc")
        )
        order = deadlines["gameweek"].to_list()
        weeks = []
        models = None
        for i, (gw, deadline) in enumerate(deadlines.iter_rows()):
            if models is None or i % self.refit_every == 0:
                models = self._fit(deadline)
            pool = self._pool(season, deadline)
            xp, upside = self._expected_points(
                season, order[i : i + self.horizon], deadline, pool, models
            )
            truth = self.gameweeks.filter((pl.col("season") == season) & (pl.col("gameweek") == gw))
            weeks.append(
                Week(
                    gameweek=gw,
                    pool=pool,
                    xp=xp,
                    upside=upside,
                    actual=dict(zip(truth["player_code"], truth["points"], strict=True)),
                    minutes=dict(zip(truth["player_code"], truth["minutes"], strict=True)),
                )
            )
        return weeks

    def _pool(self, season: str, deadline: datetime) -> pl.DataFrame:
        """Everyone registered this season so far, latest price/club/position as of g."""
        rows = self.features.filter(
            (pl.col("season") == season) & (pl.col("deadline_utc") <= deadline)
        )
        return (
            rows.sort("deadline_utc")
            .group_by("player_code")
            .agg(pl.col("position", "team_key", "price").last())
        )

    def _fit(
        self, deadline: datetime
    ) -> tuple[player_model.FittedComponents, player_model.FittedComponents]:
        before = self.features.filter(pl.col("deadline_utc") < deadline)
        seasons = sorted(before["season"].unique().to_list())[-self.config.train_seasons :]
        train = before.filter(pl.col("season").is_in(seasons))
        defcon = before.filter(pl.col("defcon_hit").is_not_null() & (pl.col("position") != "GKP"))
        planning = replace(
            self.config, features=tuple(c for c in self.config.features if c not in CROWD_ALL)
        )
        return (
            player_model.fit_components(train, defcon, self.config),
            player_model.fit_components(train, defcon, planning),
        )

    def _expected_points(
        self,
        season: str,
        upcoming: list[int],
        deadline: datetime,
        pool: pl.DataFrame,
        models: tuple[player_model.FittedComponents, player_model.FittedComponents],
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        next_model, planning_model = models
        rows = self.features.filter(
            (pl.col("season") == season) & pl.col("gameweek").is_in(upcoming)
        )
        nxt = rows.filter(pl.col("gameweek") == upcoming[0])
        parts = [(nxt, next_model)]

        later = rows.filter(pl.col("gameweek") != upcoming[0])
        if later.height:
            # Player state: his latest row up to this deadline (never a later gameweek).
            known = self.features.filter(
                (pl.col("season") == season) & (pl.col("deadline_utc") <= deadline)
            )
            state_cols = [c for c in planning_model.features if c not in FIXTURE_CONTEXT]
            state = (
                known.sort("deadline_utc")
                .group_by("player_code")
                .agg(pl.col(c).last() for c in state_cols if c != "position_code")
            )
            later = later.drop([c for c in state.columns if c != "player_code"]).join(
                state, on="player_code", how="inner"
            )
            history = self.matches.filter(pl.col("is_played") & (pl.col("kickoff_utc") < deadline))
            fixtures = self.matches.filter(
                pl.col("match_key").is_in(later["match_key"].unique().implode())
            )
            lam = pl.concat(
                [
                    fixtures.select("match_key"),
                    team_strength.Ratings().fit(history, deadline).lambdas(fixtures),
                ],
                how="horizontal",
            )
            later = (
                later.drop("team_lambda", "opponent_lambda", "p_clean_sheet")
                .join(lam, on="match_key", how="left")
                .with_columns(
                    pl.when(pl.col("is_home"))
                    .then(pl.col("lambda_home"))
                    .otherwise(pl.col("lambda_away"))
                    .alias("team_lambda"),
                    pl.when(pl.col("is_home"))
                    .then(pl.col("lambda_away"))
                    .otherwise(pl.col("lambda_home"))
                    .alias("opponent_lambda"),
                )
                .with_columns((-pl.col("opponent_lambda")).exp().alias("p_clean_sheet"))
            )
            parts.append((later, planning_model))

        frames = []
        for frame, model in parts:
            events = model.components(frame)
            frames.append(
                frame.select("player_code", "gameweek").with_columns(
                    player_model.expected_points(frame, events)
                )
            )
            if frame is nxt:
                settings = simulator.SimulationSettings(
                    draws=2000, defcon_on=season >= player_model.DEFCON_FROM_SEASON
                )
                dist = simulator.per_player_gameweek(simulator.simulate(nxt, events, settings), nxt)
                upside = dist.select("player_code", "sim_p90", "p_haul")
        xp = (
            pl.concat(frames)
            .group_by("player_code", "gameweek")
            .agg(pl.col("expected_points").sum())
        )
        # Every pool player gets a row per gameweek (0 when blank), so the squad stays valid.
        grid = pool.select("player_code").join(pl.DataFrame({"gameweek": upcoming}), how="cross")
        xp = grid.join(xp, on=["player_code", "gameweek"], how="left").with_columns(
            pl.col("expected_points").fill_null(0.0)
        )
        return xp, upside


def apply_availability(xp: pl.DataFrame, factors: pl.DataFrame, gameweek: int) -> pl.DataFrame:
    """Scale expected points by availability factors (player_code, this_week, later)."""
    joined = xp.join(factors, on="player_code", how="left")
    factor = pl.when(pl.col("gameweek") == gameweek).then("this_week").otherwise("later")
    return joined.with_columns(pl.col("expected_points") * factor.fill_null(1.0)).select(xp.columns)


def play(  # noqa: PLR0913 - chip options are keyword-only
    season: str,
    weeks: list[Week],
    settings: optimizer.Settings,
    *,
    seed: int | None = None,
    start_noise: float = 0.3,
    wildcards: Collection[int] = (),
    wildcard_threshold: float | None = None,
    price_moves: Mapping[int, Mapping[int, float]] | None = None,
    availability: Mapping[int, pl.DataFrame] | None = None,
) -> tuple[SeasonResult, list[PathRecord]]:
    """Manage one team through the prepared season.

    With `seed`, the first squad is built from noisy expected points (each player's
    multiplied by exp(N(0, start_noise))), so different seeds give different but
    reasonable starting squads: independent paths through the same season.

    Wildcard (once per half): played in the listed `wildcards` gameweeks, or, with
    `wildcard_threshold`, at the first deadline where the wildcard plan's objective
    beats the normal plan's by at least that much.

    `price_moves` (gameweek -> player -> expected selling-value change, `prices`) is
    passed to the optimiser, which values it at `settings.value_weight`.

    `availability` (gameweek -> player_code, this_week, later; E12) applies the live
    tool's multiplier from the status at that deadline: `this_week` scales the coming
    gameweek, `later` the rest of the horizon (0 for players who left).
    """
    result, records = SeasonResult(season), []
    squad: list[int] | None = None
    bank, free_transfers = 0, 1
    paid: dict[int, int] = {}
    wildcard_used: set[int] = set()
    for week in weeks:
        pool, xp = week.pool, week.xp
        if availability is not None and week.gameweek in availability:
            xp = apply_availability(xp, availability[week.gameweek], week.gameweek)
        if squad is None and seed is not None:
            rng = np.random.default_rng(seed)
            noise = pl.DataFrame({
                "player_code": pool["player_code"],
                "_noise": np.exp(rng.normal(0, start_noise, pool.height)),
            })  # fmt: skip
            xp = (
                xp.join(noise, on="player_code")
                .with_columns(pl.col("expected_points") * pl.col("_noise"))
                .drop("_noise")
            )
        price = dict(zip(pool["player_code"], pool["price"], strict=True))
        sell = {p: selling_price(paid[p], price[p]) for p in squad or []}
        players = pool.select("player_code", "position", "team_key", "price")

        solve = partial(
            optimizer.optimise, players, xp, current_squad=squad, bank=bank,
            free_transfers=free_transfers, selling_prices=sell, settings=settings,
            price_moves=(price_moves or {}).get(week.gameweek),
        )  # fmt: skip

        half = 1 if week.gameweek <= FIRST_HALF_END else 2
        can_wildcard = squad is not None and half not in wildcard_used
        chip = None
        if can_wildcard and week.gameweek in wildcards:
            chip, plan = "wildcard", solve(wildcard=True)
        else:
            plan = solve()
            if can_wildcard and wildcard_threshold is not None:
                rebuilt = solve(wildcard=True)
                if rebuilt.objective - plan.objective >= wildcard_threshold:
                    chip, plan = "wildcard", rebuilt
        if chip:
            wildcard_used.add(half)
        step = plan.gameweeks[0]
        if squad is None:
            bank = optimizer.BUDGET - sum(price[p] for p in step.squad)
            paid = {p: price[p] for p in step.squad}
        else:
            bank += sum(sell[p] for p in step.transfers_out) - sum(
                price[p] for p in step.transfers_in
            )
            for p in step.transfers_out:
                paid.pop(p)
            paid |= {p: price[p] for p in step.transfers_in}
            used = len(step.transfers_in) - step.hits
            if chip is None:  # a wildcard keeps banked transfers and earns none
                free_transfers = min(optimizer.MAX_FREE_TRANSFERS, free_transfers - used + 1)

        position = dict(zip(pool["player_code"], pool["position"], strict=True))
        this_week = {p: v for p, g, v in week.xp.iter_rows() if g == week.gameweek}
        bench = sorted(
            (p for p in step.squad if p not in step.lineup),
            key=lambda p: (position[p] == "GKP", -this_week.get(p, 0.0)),
        )
        points = score(step, bench, week.actual, week.minutes, position)
        points -= optimizer.HIT_COST * step.hits
        result.points.append(points)
        result.hits.append(step.hits)
        records.append(
            PathRecord(
                week.gameweek,
                step.squad,
                step.lineup,
                bench,
                step.captain,
                step.vice_captain,
                step.hits,
                points,
                chip,
            )
        )
        squad = step.squad
    return result, records


def main() -> None:
    """Simulate seasons for named experiment configs (tuning seasons only).

    uv run python -m fpl_models.season_sim baseline r4_xg_1 --seasons 2021-22 2022-23
    """
    import argparse  # noqa: PLC0415 - CLI only
    import json  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    from fpl_models import experiments, player_features  # noqa: PLC0415
    from fpl_models.data import DEFAULT_WAREHOUSE, load_player_matches, player_gameweeks  # noqa: PLC0415

    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="+")
    parser.add_argument("--seasons", nargs="+", default=list(experiments.TUNING_SEASONS))
    parser.add_argument("--hit-penalty", type=float, default=float(optimizer.HIT_COST))
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    matches = team_strength.load_matches(DEFAULT_WAREHOUSE)
    features = player_features.build_features(
        player_features.load_fixture_rows(DEFAULT_WAREHOUSE),
        player_features.match_lambdas(matches),
        player_features.schedule_features(matches),
    )
    truth = player_gameweeks(load_player_matches(DEFAULT_WAREHOUSE))
    out = Path("data/experiments/season_sim")
    out.mkdir(parents=True, exist_ok=True)
    for name in args.configs:
        settings = optimizer.Settings(time_limit_seconds=30, hit_penalty=args.hit_penalty)
        sim = SeasonSimulator(
            features,
            truth,
            matches,
            experiments.EXPERIMENTS[name],
            horizon=args.horizon,
            settings=settings,
        )
        label = f"{name}{args.tag}"
        for season in args.seasons:
            result = sim.run(season)
            (out / f"{label}_{season}.json").write_text(
                json.dumps({"points": result.points, "hits": result.hits})
            )
            summary = f"{label:<20} {season}: {result.total} points, {sum(result.hits)} hits"
            print(summary, flush=True)


if __name__ == "__main__":
    main()
