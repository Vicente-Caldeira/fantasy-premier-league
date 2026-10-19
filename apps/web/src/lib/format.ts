import type { Club, ClubFixture, Position } from "./types";

export const POSITIONS: Position[] = ["GKP", "DEF", "MID", "FWD"];

export const SQUAD_SHAPE: Record<Position, number> = { GKP: 2, DEF: 5, MID: 5, FWD: 3 };

export const photoUrl = (code: number) =>
  `https://resources.premierleague.com/premierleague/photos/players/110x140/p${code}.png`;

export const badgeUrl = (fplTeamCode: number) =>
  `https://resources.premierleague.com/premierleague/badges/70/t${fplTeamCode}.png`;

export const pct = (p: number | null | undefined, digits = 0) =>
  p == null ? "–" : `${(p * 100).toFixed(digits)}%`;

export const money = (m: number) => `£${m.toFixed(1)}m`;

export const signed = (x: number, digits = 1) => `${x > 0 ? "+" : x < 0 ? "−" : "±"}${Math.abs(x).toFixed(digits)}`;

export const compact = (n: number) =>
  Math.abs(n) >= 1_000_000
    ? `${(n / 1_000_000).toFixed(1)}M`
    : Math.abs(n) >= 1000
      ? `${(n / 1000).toFixed(0)}k`
      : `${n}`;

/** Difficulty 1 (easiest) .. 5, from the model rather than FPL's coarse FDR:
 *  attackers care about expected goals for, defenders about clean-sheet odds. */
export type Lens = "attack" | "defence";

export function difficulty(f: ClubFixture, lens: Lens): 1 | 2 | 3 | 4 | 5 {
  if (lens === "attack") {
    const g = f.goals_for;
    return g >= 2.0 ? 1 : g >= 1.6 ? 2 : g >= 1.25 ? 3 : g >= 0.95 ? 4 : 5;
  }
  const cs = f.clean_sheet;
  return cs >= 0.45 ? 1 : cs >= 0.35 ? 2 : cs >= 0.26 ? 3 : cs >= 0.18 ? 4 : 5;
}

export const lensFor = (position: Position): Lens =>
  position === "GKP" || position === "DEF" ? "defence" : "attack";

export const difficultyStyle = (d: number) => ({
  background: `var(--d${d})`,
  color: d === 3 ? "var(--text)" : "var(--d-text-strong)",
});

/** A club's three-letter abbreviation, always in capitals. */
export const abbr = (clubs: Record<string, Pick<Club, "short_name">>, key: string) =>
  (clubs[key]?.short_name ?? key.slice(0, 3)).toUpperCase();

export const venue = (isHome: boolean) => (isHome ? "H" : "A");

export function clubMap(clubs: Club[]) {
  return Object.fromEntries(clubs.map((c) => [c.key, c])) as Record<string, Club>;
}

/** Fixtures of a club grouped by gameweek (blank = [], double = two). */
export function fixturesByGameweek(club: Club | undefined, gameweeks: number[]) {
  return gameweeks.map((gw) => club?.fixtures.filter((f) => f.gameweek === gw) ?? []);
}

export const STATUS_LABEL: Record<string, string> = {
  a: "Available",
  d: "Doubtful",
  i: "Injured",
  s: "Suspended",
  u: "Unavailable",
  n: "Not in squad",
};
