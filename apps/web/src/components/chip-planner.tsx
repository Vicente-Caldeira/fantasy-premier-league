"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";

import { getJson, post, storedEntry } from "@/lib/client";
import { abbr, clubMap, signed } from "@/lib/format";
import type { Club, Entry, Meta, Plan, Player } from "@/lib/types";

const FIRST_HALF_END = 19;
const LAST_GW = 38;

type ChipKey = "wildcard" | "freehit" | "bboost" | "3xc";

const CHIPS: { key: ChipKey; name: string; short: string }[] = [
  { key: "3xc", name: "Triple captain", short: "TC" },
  { key: "bboost", name: "Bench boost", short: "BB" },
  { key: "freehit", name: "Free hit", short: "FH" },
  { key: "wildcard", name: "Wildcard", short: "WC" },
];

/** "bboost (GW2)" -> { key: "bboost", gameweek: 2 } */
function parseChips(used: string[]) {
  return used.flatMap((label) => {
    const m = /^(\w+) \(GW(\d+)\)$/.exec(label);
    return m ? [{ key: m[1] as ChipKey, gameweek: Number(m[2]) }] : [];
  });
}

const halfOf = (gw: number) => (gw <= FIRST_HALF_END ? 1 : 2);

function ChipStatus({ entry }: { entry: Entry }) {
  const used = parseChips(entry.chips_used);
  const now = entry.next_gameweek;
  return (
    <div className="grid gap-3 sm:grid-cols-4">
      {CHIPS.map((chip) => (
        <div key={chip.key} className="card p-4">
          <div className="text-sm font-semibold">{chip.name}</div>
          <div className="mt-2 space-y-1 text-xs">
            {[1, 2].map((half) => {
              const use = used.find((u) => u.key === chip.key && halfOf(u.gameweek) === half);
              const expired = half === 1 && now > FIRST_HALF_END;
              const range = half === 1 ? `GW1–${FIRST_HALF_END}` : `GW${FIRST_HALF_END + 1}–${LAST_GW}`;
              return (
                <div key={half} className="flex items-center justify-between gap-2">
                  <span className="text-muted">{range}</span>
                  {use ? (
                    <span className="text-faint">used GW{use.gameweek}</span>
                  ) : expired ? (
                    <span className="text-faint">expired</span>
                  ) : (
                    <span className="rounded bg-accent-soft px-1.5 py-0.5 font-medium text-accent-text">
                      {half === halfOf(now) ? "available" : "from GW20"}
                    </span>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      ))}
    </div>
  );
}

/** The squad's fixture count per gameweek to the end of the season: blanks and doubles. */
function Calendar({ squad, byCode, clubs, from }: { squad: number[]; byCode: Map<number, Player>; clubs: Club[]; from: number }) {
  const byKey = clubMap(clubs);
  const weeks = Array.from({ length: LAST_GW - from + 1 }, (_, i) => from + i);
  const count = (team: string, gw: number) => byKey[team]?.fixtures.filter((f) => f.gameweek === gw).length ?? 0;
  const league = weeks.map((gw) => ({
    blanks: clubs.filter((c) => count(c.key, gw) === 0).length,
    doubles: clubs.filter((c) => count(c.key, gw) >= 2).length,
  }));
  const players = squad.map((c) => byCode.get(c)).filter((p): p is Player => !!p);
  const playing = weeks.map((gw) => players.reduce((n, p) => n + Math.min(count(p.team_key, gw), 2), 0));
  // Per-player rows only matter once a blank or double touches the squad.
  const irregular = players.some((p) => weeks.some((gw) => count(p.team_key, gw) !== 1));
  const cell = (n: number) =>
    n === 0 ? "bg-danger/70 text-white" : n >= 2 ? "bg-accent text-black" : "bg-surface-2 text-faint";

  return (
    <div className="card overflow-hidden">
      <div className="flex items-baseline justify-between p-4 pb-2">
        <h2 className="text-sm font-semibold">Season calendar</h2>
        <span className="text-xs text-muted">
          <span className="mr-1 inline-block h-2 w-2 rounded-sm bg-danger/70" />
          blank
          <span className="mr-1 ml-3 inline-block h-2 w-2 rounded-sm bg-accent" />
          double
        </span>
      </div>
      <div className="scroll-thin overflow-x-auto px-4 pb-4">
        <table className="border-separate border-spacing-0.5 text-[10px]">
          <thead>
            <tr>
              <th className="sticky left-0 z-10 min-w-36 bg-surface" />
              {weeks.map((gw) => (
                <th key={gw} className="num w-6 font-normal text-faint">
                  {gw}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="label sticky left-0 z-10 bg-surface pr-2">League</td>
              {league.map((l, i) => (
                <td
                  key={weeks[i]}
                  title={`${l.blanks} clubs blank, ${l.doubles} clubs double`}
                  className={`num h-5 rounded-sm text-center ${l.doubles ? "bg-accent/60 text-black" : l.blanks ? "bg-danger/50 text-white" : ""}`}
                >
                  {l.doubles ? `+${l.doubles}` : l.blanks ? `−${l.blanks}` : ""}
                </td>
              ))}
            </tr>
            <tr>
              <td className="label sticky left-0 z-10 bg-surface pr-2">Your fixtures</td>
              {playing.map((n, i) => (
                <td key={weeks[i]} className={`num h-5 rounded-sm text-center font-semibold ${n !== 15 ? (n < 15 ? "text-danger" : "text-accent-text") : "text-faint"}`}>
                  {n}
                </td>
              ))}
            </tr>
            {(irregular ? players : []).map((p) => (
              <tr key={p.code}>
                <td className="sticky left-0 z-10 truncate bg-surface pr-2 text-[11px]">
                  {p.name} <span className="text-faint">{abbr(byKey, p.team_key)}</span>
                </td>
                {weeks.map((gw) => {
                  const n = count(p.team_key, gw);
                  return <td key={gw} className={`h-4 rounded-sm ${cell(n)}`} title={`GW${gw}: ${n} fixture${n === 1 ? "" : "s"}`} />;
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="border-t border-border px-4 py-2 text-[11px] text-faint">
        {irregular ? "" : "No blanks or doubles scheduled for your squad yet. "}
        From the published fixture list: blanks and doubles appear as fixtures are rescheduled for cup ties, so later weeks firm up during the season.
      </p>
    </div>
  );
}

export function ChipPlanner({ players, clubs, meta }: { players: Player[]; clubs: Club[]; meta: Meta }) {
  const byCode = useMemo(() => new Map(players.map((p) => [p.code, p])), [players]);
  const [entry, setEntry] = useState<Entry | null>(null);
  const [missing, setMissing] = useState(false);
  const [evaluation, setEvaluation] = useState<Plan | null>(null);
  const [freeHit, setFreeHit] = useState<Record<number, number>>({});
  const [wildcard, setWildcard] = useState<{ gain: number } | "running" | null>(null);
  const horizon = meta.horizon;

  useEffect(() => {
    const id = storedEntry();
    if (id == null) {
      const t = setTimeout(() => setMissing(true), 0);
      return () => clearTimeout(t);
    }
    getJson<Entry>(`/entry/${id}`)
      .then(setEntry)
      .catch(() => setMissing(true));
  }, []);

  useEffect(() => {
    if (!entry) return;
    let cancelled = false;
    (async () => {
      const ev = await post<Plan>("/squad/evaluate", { squad: entry.squad, horizon: horizon.length });
      if (cancelled) return;
      setEvaluation(ev);
      // One small solve per gameweek: the best one-week squad within the team's value.
      for (const [i, gw] of horizon.entries()) {
        const plan = await post<Plan>("/freehit", {
          squad: entry.squad,
          horizon: 1,
          gameweek: gw,
          bank: entry.bank,
          selling_prices: entry.selling_prices,
        }).catch(() => null);
        if (cancelled) return;
        if (plan) setFreeHit((current) => ({ ...current, [gw]: plan.gameweeks[0].xp - ev.gameweeks[i].xp }));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [entry, horizon]);

  const estimateWildcard = async () => {
    if (!entry) return;
    setWildcard("running");
    const body = {
      squad: entry.squad,
      horizon: horizon.length,
      bank: entry.bank,
      free_transfers: entry.free_transfers,
      selling_prices: entry.selling_prices,
    };
    const [normal, wc] = await Promise.all([post<Plan>("/plan", body), post<Plan>("/plan", { ...body, wildcard: true })]);
    setWildcard({ gain: wc.total_xp - normal.total_xp });
  };

  if (missing) {
    return (
      <p className="py-16 text-center text-sm text-muted">
        Load your team on{" "}
        <Link href="/" className="text-accent-text hover:underline">
          My team
        </Link>{" "}
        first; the chip planner uses the same team.
      </p>
    );
  }
  if (!entry) return <div className="h-96 animate-pulse rounded-xl bg-surface-2" />;

  const used = parseChips(entry.chips_used);
  const available = (key: ChipKey) =>
    !used.some((u) => u.key === key && halfOf(u.gameweek) === halfOf(entry.next_gameweek));

  const values = horizon.map((gw, i) => {
    const week = evaluation?.gameweeks[i];
    const captain = week ? byCode.get(week.captain) : undefined;
    const bench = week
      ? entry.squad.filter((c) => !week.lineup.includes(c)).reduce((s, c) => s + (byCode.get(c)?.xp[i] ?? 0), 0)
      : null;
    return { gw, tc: captain ? captain.xp[i] : null, captain, bb: bench, fh: freeHit[gw] ?? null };
  });
  const bestOf = (key: "tc" | "bb" | "fh") =>
    values.reduce<number | null>((best, v) => (v[key] != null && (best == null || v[key]! > best) ? v[key] : best), null);

  const advice: Record<ChipKey, string> = {
    bboost:
      "Play it when your bench is expected to score 10+ — usually a double gameweek. This rule held up on seasons the model never saw (+6 a season over playing it blind).",
    "3xc":
      "Prefer a double gameweek with a captain on 10+ xP. On unseen seasons that rule didn't beat playing it blind, so treat it as a preference, not a rule.",
    freehit:
      "In our tests a free hit was worth about the same wherever it was played. Its real use is a blank gameweek where several of your players don't play.",
    wildcard:
      "Our 'rebuild when it gains 20+ xP' rule looked strong in tuning (+72) but only gained +10 a season on unseen seasons. Use it when injuries, price drops or a fixture swing pile up.",
  };

  return (
    <div className="space-y-6">
      <ChipStatus entry={entry} />

      <div className="card overflow-hidden">
        <div className="flex items-baseline justify-between p-4 pb-2">
          <h2 className="text-sm font-semibold">Chip value by gameweek</h2>
          <span className="text-xs text-muted">expected extra points for your current squad</span>
        </div>
        <div className="scroll-thin overflow-x-auto">
          <table className="w-full min-w-[640px] text-sm">
            <thead>
              <tr className="border-b border-border text-right">
                <th className="label py-2 pl-4 text-left font-medium">Chip</th>
                {horizon.map((gw) => (
                  <th key={gw} className="label py-2 pr-4 font-medium">
                    GW{gw}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {(
                [
                  { key: "3xc", label: "Triple captain", field: "tc", hint: (v: (typeof values)[number]) => v.captain?.name },
                  { key: "bboost", label: "Bench boost", field: "bb", hint: () => undefined },
                  { key: "freehit", label: "Free hit", field: "fh", hint: () => undefined },
                ] as const
              ).map((row) => {
                const best = bestOf(row.field);
                return (
                  <tr key={row.key} className={`border-b border-border/60 text-right ${available(row.key) ? "" : "opacity-45"}`}>
                    <td className="py-2.5 pl-4 text-left">
                      {row.label}
                      {!available(row.key) ? <span className="ml-2 text-xs text-faint">used this half</span> : null}
                    </td>
                    {values.map((v) => {
                      const x = v[row.field];
                      const top = x != null && best != null && x === best;
                      return (
                        <td key={v.gw} className="py-2.5 pr-4">
                          {x == null ? (
                            <span className="text-faint">…</span>
                          ) : (
                            <span className={`num inline-block rounded px-1.5 ${top ? "bg-accent-soft font-semibold text-accent-text" : ""}`}>
                              {signed(x)}
                            </span>
                          )}
                          {row.hint(v) ? <div className="text-[10px] text-faint">{row.hint(v)}</div> : null}
                        </td>
                      );
                    })}
                  </tr>
                );
              })}
              <tr className={`text-right ${available("wildcard") ? "" : "opacity-45"}`}>
                <td className="py-2.5 pl-4 text-left">
                  Wildcard
                  {!available("wildcard") ? <span className="ml-2 text-xs text-faint">used this half</span> : null}
                </td>
                <td colSpan={horizon.length} className="py-2.5 pr-4">
                  {wildcard === null ? (
                    <button onClick={estimateWildcard} className="rounded-md border border-border px-2.5 py-1 text-xs hover:border-accent">
                      Estimate over GW{horizon[0]}–{horizon[horizon.length - 1]} (~40 s)
                    </button>
                  ) : wildcard === "running" ? (
                    <span className="text-xs text-muted">Solving both plans…</span>
                  ) : (
                    <span className="num">
                      {signed(wildcard.gain)} xP <span className="text-xs text-faint">vs the best plan without it, hits included</span>
                    </span>
                  )}
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <div className="grid gap-3 sm:grid-cols-2">
        {CHIPS.map((chip) => (
          <div key={chip.key} className="card flex gap-3 p-4">
            <span className="grid h-8 w-8 shrink-0 place-items-center rounded-md bg-surface-2 text-[11px] font-bold">{chip.short}</span>
            <div>
              <div className="text-sm font-semibold">{chip.name}</div>
              <p className="mt-0.5 text-xs leading-relaxed text-muted">{advice[chip.key]}</p>
            </div>
          </div>
        ))}
      </div>

      <Calendar squad={entry.squad} byCode={byCode} clubs={clubs} from={entry.next_gameweek} />

      <p className="text-xs text-faint">Values assume today&apos;s squad; transfers change them.</p>
    </div>
  );
}
