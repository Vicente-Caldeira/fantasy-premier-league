"use client";

import Link from "next/link";
import { useMemo, useState } from "react";

import { ClubBadge, OutcomeBar } from "@/components/ui";
import { abbr, clubMap, difficultyStyle, pct, venue } from "@/lib/format";
import type { Club, ClubFixture, Fixture } from "@/lib/types";

type Mode = "attack" | "defence" | "win";

const MODES: { key: Mode; label: string; hint: string }[] = [
  { key: "attack", label: "Attack", hint: "expected goals for — pick attackers" },
  { key: "defence", label: "Defence", hint: "clean-sheet chance — pick defenders and keepers" },
  { key: "win", label: "Win", hint: "chance of winning" },
];

const value = (f: ClubFixture, mode: Mode) =>
  mode === "attack" ? f.goals_for : mode === "defence" ? f.clean_sheet : f.win;

/** 1 (best) .. 5 from a value, with thresholds per mode. */
function grade(v: number, mode: Mode): number {
  const t = mode === "attack" ? [2.0, 1.6, 1.25, 0.95] : mode === "defence" ? [0.45, 0.35, 0.26, 0.18] : [0.62, 0.5, 0.38, 0.26];
  return v >= t[0] ? 1 : v >= t[1] ? 2 : v >= t[2] ? 3 : v >= t[3] ? 4 : 5;
}

const fmt = (v: number, mode: Mode) => (mode === "attack" ? v.toFixed(1) : `${Math.round(v * 100)}`);

export function FixtureTicker({ clubs, fixtures, gameweeks }: { clubs: Club[]; fixtures: Fixture[]; gameweeks: number[] }) {
  const byKey = useMemo(() => clubMap(clubs), [clubs]);
  const [mode, setMode] = useState<Mode>("attack");
  const [span, setSpan] = useState(6);
  const [showValues, setShowValues] = useState(true);
  const [matchWeek, setMatchWeek] = useState(gameweeks[0]);
  const weeks = gameweeks.slice(0, span);

  const rows = useMemo(
    () =>
      clubs
        .map((club) => {
          const cells = weeks.map((gw) => club.fixtures.filter((f) => f.gameweek === gw));
          const all = cells.flat();
          // Sum over the window: a double gameweek counts twice, a blank as nothing.
          const total = all.reduce((s, f) => s + value(f, mode), 0);
          return { club, cells, total };
        })
        .sort((a, b) => b.total - a.total),
    [clubs, weeks, mode],
  );

  const matches = fixtures.filter((f) => f.gameweek === matchWeek && f.home_goals == null);

  return (
    <div className="space-y-6">
      <div className="card overflow-hidden">
        <div className="flex flex-wrap items-center gap-3 border-b border-border p-3">
          <div className="flex rounded-lg bg-surface-2 p-0.5 text-sm">
            {MODES.map((m) => (
              <button
                key={m.key}
                onClick={() => setMode(m.key)}
                className={`rounded-md px-3 py-1 ${mode === m.key ? "bg-surface font-medium shadow-sm" : "text-muted hover:text-text"}`}
              >
                {m.label}
              </button>
            ))}
          </div>
          <span className="text-xs text-muted">{MODES.find((m) => m.key === mode)?.hint}</span>
          <div className="ml-auto flex items-center gap-3 text-sm">
            <label className="flex items-center gap-1.5 text-muted">
              <input type="checkbox" checked={showValues} onChange={(e) => setShowValues(e.target.checked)} className="accent-[var(--accent)]" />
              values
            </label>
            <select
              value={span}
              onChange={(e) => setSpan(Number(e.target.value))}
              className="rounded-lg border border-border bg-surface px-2 py-1.5"
              aria-label="Gameweeks"
            >
              {[3, 4, 6, 8, 10].filter((n) => n <= gameweeks.length).map((n) => (
                <option key={n} value={n}>
                  Next {n} GWs
                </option>
              ))}
            </select>
          </div>
        </div>
        <div className="scroll-thin overflow-x-auto">
          <table className="w-full min-w-[760px] border-separate border-spacing-1 px-2 py-2 text-sm">
            <thead>
              <tr>
                <th className="label w-40 text-left font-medium">Club</th>
                {weeks.map((gw) => (
                  <th key={gw} className="label font-medium">
                    GW{gw}
                  </th>
                ))}
                <th className="label w-16 text-right font-medium" title="Sum over the window (doubles count twice)">
                  {mode === "attack" ? "xG Σ" : mode === "defence" ? "CS Σ" : "Wins Σ"}
                </th>
              </tr>
            </thead>
            <tbody>
              {rows.map(({ club, cells, total }) => (
                <tr key={club.key}>
                  <td>
                    <Link href={`/clubs/${club.key}`} className="flex items-center gap-2 hover:text-accent-text">
                      <ClubBadge club={club} size={18} />
                      <span className="truncate">{club.name}</span>
                    </Link>
                  </td>
                  {cells.map((fs, i) => (
                    <td key={weeks[i]} className="p-0">
                      {fs.length === 0 ? (
                        <div className="flex h-9 items-center justify-center rounded-md text-xs text-faint ring-1 ring-border ring-inset">
                          blank
                        </div>
                      ) : (
                        <div className="flex h-9 gap-px overflow-hidden rounded-md">
                          {fs.map((f) => {
                            const v = value(f, mode);
                            const opp = byKey[f.opponent];
                            return (
                              <div
                                key={f.opponent}
                                style={difficultyStyle(grade(v, mode))}
                                className="flex flex-1 flex-col items-center justify-center leading-none"
                                title={`${opp?.name} (${f.is_home ? "H" : "A"}) · xG ${f.goals_for.toFixed(2)}–${f.goals_against.toFixed(2)} · CS ${pct(f.clean_sheet)} · W ${pct(f.win)}`}
                              >
                                <span className="text-[11px] font-semibold whitespace-nowrap">
                                  {abbr(byKey, f.opponent)}
                                  <span className="ml-0.5 text-[0.85em] font-normal opacity-75">({venue(f.is_home)})</span>
                                </span>
                                {showValues ? <span className="num mt-0.5 text-[10px] opacity-80">{fmt(v, mode)}</span> : null}
                              </div>
                            );
                          })}
                        </div>
                      )}
                    </td>
                  ))}
                  <td className="num text-right font-semibold">{mode === "attack" ? total.toFixed(1) : total.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="border-t border-border px-3 py-2 text-xs text-faint">
          From the team model: bookmaker odds where priced, attack/defence ratings beyond. (H) home, (A) away.
        </p>
      </div>

      <div>
        <div className="mb-3 flex flex-wrap items-center gap-2">
          <h2 className="text-sm font-semibold">Matches</h2>
          <div className="flex flex-wrap gap-1">
            {gameweeks.slice(0, 8).map((gw) => (
              <button
                key={gw}
                onClick={() => setMatchWeek(gw)}
                className={`rounded-md px-2 py-1 text-xs ${matchWeek === gw ? "bg-surface-2 font-medium" : "text-muted hover:text-text"}`}
              >
                GW{gw}
              </button>
            ))}
          </div>
        </div>
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {matches.map((m) => {
            const home = byKey[m.home];
            const away = byKey[m.away];
            return (
              <div key={m.match_key} className="card p-4">
                <div className="flex items-center justify-between text-xs text-muted">
                  <span>
                    {new Date(m.kickoff_utc).toLocaleString(undefined, { weekday: "short", day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" })}
                  </span>
                  <span className={`rounded px-1.5 py-0.5 text-[10px] ${m.source === "market" ? "bg-accent-soft text-accent-text" : "bg-surface-2"}`}>
                    {m.source === "market" ? "odds" : "ratings"}
                  </span>
                </div>
                <div className="mt-3 grid grid-cols-[1fr_auto_1fr] items-center gap-2">
                  <Link href={`/clubs/${m.home}`} className="flex items-center gap-2 font-medium hover:text-accent-text">
                    <ClubBadge club={home} size={22} />
                    <span className="truncate">{abbr(byKey, m.home)}</span>
                  </Link>
                  <span className="num text-sm text-muted">
                    {m.lambda_home?.toFixed(1)} – {m.lambda_away?.toFixed(1)}
                  </span>
                  <Link href={`/clubs/${m.away}`} className="flex items-center justify-end gap-2 font-medium hover:text-accent-text">
                    <span className="truncate">{abbr(byKey, m.away)}</span>
                    <ClubBadge club={away} size={22} />
                  </Link>
                </div>
                <div className="mt-3">
                  <OutcomeBar win={m.home_win ?? 0} draw={m.draw ?? 0} loss={m.away_win ?? 0} />
                  <div className="num mt-1 flex justify-between text-[11px] text-muted">
                    <span>{pct(m.home_win)}</span>
                    <span>draw {pct(m.draw)}</span>
                    <span>{pct(m.away_win)}</span>
                  </div>
                </div>
                <div className="num mt-3 grid grid-cols-3 border-t border-border pt-2 text-center text-[11px]">
                  <span>
                    <span className="block text-muted">CS {abbr(byKey, m.home)}</span>
                    {pct(m.home_clean_sheet)}
                  </span>
                  <span>
                    <span className="block text-muted">Over 2.5</span>
                    {pct(m.over_2_5)}
                  </span>
                  <span>
                    <span className="block text-muted">CS {abbr(byKey, m.away)}</span>
                    {pct(m.away_clean_sheet)}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
