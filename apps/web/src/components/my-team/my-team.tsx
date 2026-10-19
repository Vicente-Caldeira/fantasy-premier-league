"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";

import { Pitch, type PitchSlot } from "@/components/my-team/pitch";
import { PlayerPhoto } from "@/components/photo";
import { ClubBadge, FixtureCell, PriceMove, StatusPill } from "@/components/ui";
import { abbr, clubMap, compact, lensFor, money, pct, signed } from "@/lib/format";
import { ENTRY_KEY, post } from "@/lib/client";
import type { Club, Entry, Meta, Plan, Player } from "@/lib/types";

const HIT = 4;

interface Swap {
  out: number;
  in: number;
}

/** Pair a gameweek's sales with purchases of the same position. */
function pairTransfers(outs: number[], ins: number[], byCode: Map<number, Player>): Swap[] {
  const remaining = [...outs];
  const pairs: Swap[] = [];
  for (const incoming of ins) {
    const position = byCode.get(incoming)?.position;
    const i = remaining.findIndex((o) => byCode.get(o)?.position === position);
    if (i >= 0) pairs.push({ out: remaining.splice(i, 1)[0], in: incoming });
  }
  return pairs;
}


function EntryForm({ onSubmit, error, loading }: { onSubmit: (id: number) => void; error: string | null; loading: boolean }) {
  const [value, setValue] = useState("");
  return (
    <div className="mx-auto max-w-lg py-16 text-center">
      <h1 className="text-2xl font-semibold tracking-tight">Load your FPL team</h1>
      <p className="mt-2 text-sm text-muted">
        See expected points for your squad, try transfers before you make them, and get the optimiser&apos;s
        suggestions. Your team ID is in the URL of your Points page:{" "}
        <span className="num text-text">fantasy.premierleague.com/entry/<b>1234567</b>/event/7</span>. It&apos;s
        kept in this browser only.
      </p>
      <form
        className="mt-6 flex gap-2"
        onSubmit={(e) => {
          e.preventDefault();
          const id = Number(value.trim());
          if (Number.isInteger(id) && id > 0) onSubmit(id);
        }}
      >
        <input
          value={value}
          onChange={(e) => setValue(e.target.value)}
          inputMode="numeric"
          placeholder="Team ID"
          className="num flex-1 rounded-lg border border-border bg-surface px-3 py-2 outline-none focus:border-accent"
        />
        <button disabled={loading} className="rounded-lg bg-accent px-4 py-2 font-medium text-black disabled:opacity-60">
          {loading ? "Loading…" : "Load team"}
        </button>
      </form>
      {error ? <p className="mt-3 text-sm text-danger">{error}</p> : null}
    </div>
  );
}

export function MyTeam({ players, clubs, meta }: { players: Player[]; clubs: Club[]; meta: Meta }) {
  const byCode = useMemo(() => new Map(players.map((p) => [p.code, p])), [players]);
  const byClub = useMemo(() => clubMap(clubs), [clubs]);
  const maxHorizon = meta.horizon.length;

  const [entryId, setEntryId] = useState<number | null>(null);
  const [entry, setEntry] = useState<Entry | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [swaps, setSwaps] = useState<Swap[]>([]);
  const [horizon, setHorizon] = useState(Math.min(5, maxHorizon));
  const [gwIndex, setGwIndex] = useState(0);
  const [selected, setSelected] = useState<number | null>(null);
  const [replacing, setReplacing] = useState(false);
  const [evaluation, setEvaluation] = useState<Plan | null>(null);
  const [baseline, setBaseline] = useState<Plan | null>(null);
  const [plan, setPlan] = useState<Plan | null>(null);
  const [planning, setPlanning] = useState(false);
  const [planError, setPlanError] = useState<string | null>(null);

  const load = useCallback(async (id: number) => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`/api/entry/${id}`);
      if (!response.ok) throw new Error(response.status === 502 ? "FPL didn't recognise that team ID." : response.statusText);
      const data = (await response.json()) as Entry;
      setEntry(data);
      setEntryId(id);
      setSwaps([]);
      setPlan(null);
      try {
        localStorage.setItem(ENTRY_KEY, String(id));
      } catch {}
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    let stored: string | null = null;
    try {
      stored = localStorage.getItem(ENTRY_KEY);
    } catch {}
    // eslint-disable-next-line react-hooks/set-state-in-effect -- restoring a saved team once on mount
    if (stored) void load(Number(stored));
  }, [load]);

  const original = useMemo(() => entry?.squad ?? [], [entry]);
  const squad = useMemo(
    () => original.map((c) => swaps.find((s) => s.out === c)?.in ?? c),
    [original, swaps],
  );
  const sellPrice = useCallback(
    (code: number) => entry?.selling_prices[String(code)] ?? byCode.get(code)?.price ?? 0,
    [entry, byCode],
  );
  const bank = useMemo(() => {
    if (!entry) return 0;
    const out = swaps.reduce((sum, s) => sum + sellPrice(s.out), 0);
    const spent = swaps.reduce((sum, s) => sum + (byCode.get(s.in)?.price ?? 0), 0);
    return Math.round((entry.bank + out - spent) * 10) / 10;
  }, [entry, swaps, sellPrice, byCode]);
  const transfers = swaps.length;
  const hits = entry ? Math.max(0, transfers - entry.free_transfers) : 0;

  // Best XI and captain for the current (sandbox) squad and for the untouched squad.
  useEffect(() => {
    if (squad.length !== 15) return;
    let cancelled = false;
    const id = setTimeout(() => {
      post<Plan>("/squad/evaluate", { squad, horizon })
        .then((p) => !cancelled && setEvaluation(p))
        .catch(() => !cancelled && setEvaluation(null));
    }, 120);
    return () => {
      cancelled = true;
      clearTimeout(id);
    };
  }, [squad, horizon]);
  useEffect(() => {
    if (original.length !== 15) return;
    let cancelled = false;
    post<Plan>("/squad/evaluate", { squad: original, horizon })
      .then((p) => !cancelled && setBaseline(p))
      .catch(() => !cancelled && setBaseline(null));
    return () => {
      cancelled = true;
    };
  }, [original, horizon]);

  const week = evaluation?.gameweeks[Math.min(gwIndex, (evaluation?.gameweeks.length ?? 1) - 1)];
  const gameweek = meta.horizon[gwIndex] ?? meta.horizon[0];
  const xpFor = useCallback((p: Player, i: number) => p.xp[i] ?? 0, []);

  const slots = useMemo(() => {
    const make = (code: number): PitchSlot | null => {
      const player = byCode.get(code);
      if (!player) return null;
      return {
        player,
        xp: xpFor(player, gwIndex),
        captain: week?.captain === code,
        vice: week?.vice_captain === code,
        isNew: !original.includes(code),
      };
    };
    const lineupCodes = week?.lineup ?? entry?.lineup ?? squad.slice(0, 11);
    const lineup = lineupCodes.map(make).filter((s): s is PitchSlot => s !== null);
    const bench = squad
      .filter((c) => !lineupCodes.includes(c))
      .map(make)
      .filter((s): s is PitchSlot => s !== null)
      .sort((a, b) => Number(b.player.position === "GKP") - Number(a.player.position === "GKP") || b.xp - a.xp);
    return { lineup, bench };
  }, [byCode, xpFor, gwIndex, week, original, entry, squad]);

  const horizonXp = (evaluation?.total_xp ?? 0) - HIT * hits;
  const delta = baseline && evaluation ? horizonXp - baseline.total_xp : null;

  const transferOut = (outCode: number, inCode: number) => {
    setSwaps((current) => {
      const existing = current.find((s) => s.in === outCode);
      if (existing) {
        // Replacing a player bought in the sandbox: keep the original sale.
        return existing.out === inCode
          ? current.filter((s) => s !== existing)
          : current.map((s) => (s === existing ? { out: s.out, in: inCode } : s));
      }
      return [...current, { out: outCode, in: inCode }];
    });
    setSelected(inCode);
    setReplacing(false);
  };
  const undo = (inCode: number) => {
    setSwaps((current) => current.filter((s) => s.in !== inCode));
    setSelected(null);
  };

  const suggest = async (wildcard = false) => {
    if (!entry) return;
    setPlanning(true);
    setPlanError(null);
    try {
      setPlan(
        await post<Plan>("/plan", {
          squad: original,
          horizon,
          bank: entry.bank,
          free_transfers: entry.free_transfers,
          selling_prices: entry.selling_prices,
          wildcard,
        }),
      );
    } catch (e) {
      setPlanError(e instanceof Error ? e.message : String(e));
    } finally {
      setPlanning(false);
    }
  };
  const applyFirstWeek = () => {
    const first = plan?.gameweeks[0];
    if (!first) return;
    setSwaps(pairTransfers(first.transfers_out, first.transfers_in, byCode));
    setSelected(null);
  };

  if (!entry) {
    return (
      <EntryForm
        onSubmit={(id) => void load(id)}
        error={error}
        loading={loading || (entryId == null && loading)}
      />
    );
  }

  const selectedPlayer = selected != null ? byCode.get(selected) : undefined;

  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_380px]">
      <div className="space-y-4">
        {/* Team header */}
        <div className="flex flex-wrap items-end justify-between gap-3">
          <div>
            <h1 className="text-xl font-semibold tracking-tight">{entry.name}</h1>
            <p className="text-sm text-muted">
              {entry.manager} · rank <span className="num text-text">{entry.overall_rank?.toLocaleString() ?? "–"}</span> ·{" "}
              <span className="num text-text">{entry.total_points}</span> pts
            </p>
          </div>
          <div className="flex items-center gap-2 text-sm">
            <label className="text-muted">Plan over</label>
            <div className="flex rounded-lg bg-surface-2 p-0.5">
              {[1, 3, 5, 8].filter((h) => h <= maxHorizon).map((h) => (
                <button
                  key={h}
                  onClick={() => setHorizon(h)}
                  className={`rounded-md px-2.5 py-1 ${horizon === h ? "bg-surface font-medium shadow-sm" : "text-muted"}`}
                >
                  {h} GW{h > 1 ? "s" : ""}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Gameweek tabs with each week's best-XI expected points */}
        <div className="grid grid-cols-5 gap-2">
          {meta.horizon.slice(0, Math.max(horizon, 1)).map((gw, i) => {
            const g = evaluation?.gameweeks[i];
            const b = baseline?.gameweeks[i];
            const diff = g && b ? g.xp - b.xp - (i === 0 ? HIT * hits : 0) : 0;
            return (
              <button
                key={gw}
                onClick={() => setGwIndex(i)}
                className={`card px-3 py-2 text-left transition ${gwIndex === i ? "ring-2 ring-accent" : "hover:border-faint"}`}
              >
                <div className="label">GW{gw}</div>
                <div className="num text-lg font-semibold">{g ? g.xp.toFixed(1) : "…"}</div>
                {Math.abs(diff) >= 0.05 ? (
                  <div className={`num text-xs ${diff > 0 ? "text-accent-text" : "text-danger"}`}>{signed(diff)}</div>
                ) : (
                  <div className="text-xs text-faint">xP</div>
                )}
              </button>
            );
          })}
        </div>

        <Pitch
          lineup={slots.lineup}
          bench={slots.bench}
          clubs={byClub}
          gameweek={gameweek}
          selected={selected}
          onSelect={(code) => {
            setSelected(code === selected ? null : code);
            setReplacing(false);
          }}
        />
        <p className="text-xs text-faint">
          Lineup and captain are re-optimised for each gameweek. Click a player to see details or try a transfer.
          Captain&apos;s points are shown doubled.
        </p>
      </div>

      {/* Side panel */}
      <aside className="space-y-4 lg:sticky lg:top-20 lg:self-start">
        <div className="card p-4">
          <div className="grid grid-cols-3 gap-3">
            <div>
              <div className="label">xP · {horizon} GW{horizon > 1 ? "s" : ""}</div>
              <div className="num mt-1 text-2xl font-semibold">{evaluation ? horizonXp.toFixed(1) : "…"}</div>
              {delta != null && transfers > 0 ? (
                <div className={`num text-xs ${delta >= 0 ? "text-accent-text" : "text-danger"}`}>{signed(delta)} vs now</div>
              ) : (
                <div className="text-xs text-faint">after hits</div>
              )}
            </div>
            <div>
              <div className="label">Transfers</div>
              <div className="num mt-1 text-2xl font-semibold">
                {transfers}
                <span className="text-sm text-muted">/{entry.free_transfers}</span>
              </div>
              <div className={`num text-xs ${hits ? "text-danger" : "text-faint"}`}>{hits ? `−${HIT * hits} hit` : "free"}</div>
            </div>
            <div>
              <div className="label">Bank</div>
              <div className={`num mt-1 text-2xl font-semibold ${bank < 0 ? "text-danger" : ""}`}>{bank.toFixed(1)}</div>
              <div className="text-xs text-faint">£m</div>
            </div>
          </div>
          {transfers > 0 ? (
            <div className="mt-4 space-y-1.5 border-t border-border pt-3">
              {swaps.map((s) => (
                <div key={s.in} className="flex items-center gap-2 text-sm">
                  <span className="truncate text-muted line-through">{byCode.get(s.out)?.name}</span>
                  <span className="text-faint">→</span>
                  <span className="truncate font-medium">{byCode.get(s.in)?.name}</span>
                  <button onClick={() => undo(s.in)} className="ml-auto text-xs text-muted hover:text-danger">
                    undo
                  </button>
                </div>
              ))}
              <button onClick={() => setSwaps([])} className="pt-1 text-xs font-medium text-muted hover:text-text">
                Reset all
              </button>
            </div>
          ) : null}
          {bank < 0 ? <p className="mt-3 text-xs text-danger">Over budget by {money(-bank)}.</p> : null}
        </div>

        {selectedPlayer ? (
          <PlayerPanel
            player={selectedPlayer}
            clubs={byClub}
            horizon={meta.horizon}
            sell={sellPrice(selectedPlayer.code)}
            isNew={!original.includes(selectedPlayer.code)}
            replacing={replacing}
            onReplace={() => setReplacing(true)}
            onUndo={() => undo(selectedPlayer.code)}
            onClose={() => setSelected(null)}
          >
            {replacing ? (
              <Replacements
                out={selectedPlayer}
                players={players}
                squad={squad}
                clubs={byClub}
                budget={bank + sellPrice(selectedPlayer.code)}
                horizon={horizon}
                gameweeks={meta.horizon}
                onPick={(code) => transferOut(selectedPlayer.code, code)}
              />
            ) : null}
          </PlayerPanel>
        ) : (
          <Planner
            plan={plan}
            planning={planning}
            error={planError}
            baseline={baseline}
            byCode={byCode}
            onSuggest={suggest}
            onApply={applyFirstWeek}
          />
        )}

        <Captaincy squad={squad} byCode={byCode} gwIndex={gwIndex} gameweek={gameweek} />

        <button
          onClick={() => {
            try {
              localStorage.removeItem(ENTRY_KEY);
            } catch {}
            setEntry(null);
            setEntryId(null);
          }}
          className="text-xs text-faint hover:text-text"
        >
          Use a different team
        </button>
      </aside>
    </div>
  );
}

function PlayerPanel({
  player: p,
  clubs,
  horizon,
  sell,
  isNew,
  replacing,
  onReplace,
  onUndo,
  onClose,
  children,
}: {
  player: Player;
  clubs: Record<string, Club>;
  horizon: number[];
  sell: number;
  isNew: boolean;
  replacing: boolean;
  onReplace: () => void;
  onUndo: () => void;
  onClose: () => void;
  children: React.ReactNode;
}) {
  const club = clubs[p.team_key];
  return (
    <div className="card overflow-hidden">
      <div className="flex items-center gap-3 p-4">
        <PlayerPhoto code={p.code} name={p.name} size={48} />
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-1.5">
            <Link href={`/players/${p.code}`} className="truncate font-semibold hover:text-accent-text">
              {p.name}
            </Link>
            <StatusPill player={p} />
          </div>
          <div className="flex items-center gap-1 text-xs text-muted">
            <ClubBadge club={club} size={12} /> {abbr(clubs, p.team_key)} · {p.position} · {money(p.price)}
            {Math.abs(sell - p.price) >= 0.05 ? <> · sells {money(sell)}</> : null}
          </div>
        </div>
        <button onClick={onClose} aria-label="Close" className="self-start text-faint hover:text-text">
          ✕
        </button>
      </div>
      {p.news ? <p className="border-y border-border bg-warn/10 px-4 py-2 text-xs">{p.news}</p> : null}
      <div className="grid grid-cols-5 gap-1 px-4 py-3">
        {horizon.map((gw, i) => (
          <div key={gw} className="text-center">
            <FixtureCell club={club} clubs={clubs} gameweek={gw} lens={lensFor(p.position)} compact />
            <div className="num mt-1 text-sm font-semibold">{(p.xp[i] ?? 0).toFixed(1)}</div>
          </div>
        ))}
      </div>
      <div className="grid grid-cols-4 gap-2 border-t border-border px-4 py-3 text-center text-xs">
        <div>
          <div className="label">Form</div>
          <div className="num mt-0.5 text-sm">{p.season.form.toFixed(1)}</div>
        </div>
        <div>
          <div className="label">Haul</div>
          <div className="num mt-0.5 text-sm">{pct(p.p_haul)}</div>
        </div>
        <div>
          <div className="label">60+</div>
          <div className="num mt-0.5 text-sm">{pct(p.p_60)}</div>
        </div>
        <div>
          <div className="label">Price</div>
          <div className="num mt-0.5 text-sm">
            <PriceMove player={p} />
          </div>
        </div>
      </div>
      {!replacing ? (
        <div className="flex gap-2 border-t border-border p-3">
          <button onClick={onReplace} className="flex-1 rounded-lg bg-accent px-3 py-2 text-sm font-medium text-black">
            Transfer out
          </button>
          {isNew ? (
            <button onClick={onUndo} className="rounded-lg border border-border px-3 py-2 text-sm">
              Undo
            </button>
          ) : null}
        </div>
      ) : null}
      {children}
    </div>
  );
}

function Replacements({
  out,
  players,
  squad,
  clubs,
  budget,
  horizon,
  gameweeks,
  onPick,
}: {
  out: Player;
  players: Player[];
  squad: number[];
  clubs: Record<string, Club>;
  budget: number;
  horizon: number;
  gameweeks: number[];
  onPick: (code: number) => void;
}) {
  const [search, setSearch] = useState("");
  const [showAll, setShowAll] = useState(false);
  const sum = (p: Player) => p.xp.slice(0, horizon).reduce((a, b) => a + b, 0);
  const base = sum(out);
  const clubCount = useMemo(() => {
    const counts: Record<string, number> = {};
    for (const code of squad) {
      const p = players.find((x) => x.code === code);
      if (p && p.code !== out.code) counts[p.team_key] = (counts[p.team_key] ?? 0) + 1;
    }
    return counts;
  }, [squad, players, out]);
  const q = search.trim().toLowerCase();
  const candidates = players
    .filter((p) => p.position === out.position && !squad.includes(p.code))
    .filter((p) => !q || p.name.toLowerCase().includes(q))
    .map((p) => ({ p, gain: sum(p) - base, affordable: p.price <= budget + 1e-9, clubOk: (clubCount[p.team_key] ?? 0) < 3 }))
    .filter((c) => showAll || (c.affordable && c.clubOk))
    .sort((a, b) => b.gain - a.gain)
    .slice(0, 40);

  return (
    <div className="border-t border-border">
      <div className="flex items-center gap-2 px-3 pt-3">
        <input
          autoFocus
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder={`Replace ${out.name}…`}
          className="flex-1 rounded-lg border border-border bg-surface px-2.5 py-1.5 text-sm outline-none focus:border-accent"
        />
        <label className="flex items-center gap-1 text-xs text-muted">
          <input type="checkbox" checked={showAll} onChange={(e) => setShowAll(e.target.checked)} className="accent-[var(--accent)]" />
          all
        </label>
      </div>
      <p className="px-3 pt-1.5 text-[11px] text-faint">
        Budget {money(budget)} · sorted by xP gain over {horizon} GW{horizon > 1 ? "s" : ""}
      </p>
      <ul className="scroll-thin mt-2 max-h-[420px] overflow-y-auto">
        {candidates.map(({ p, gain, affordable, clubOk }) => {
          const club = clubs[p.team_key];
          const disabled = !affordable || !clubOk;
          return (
            <li key={p.code}>
              <button
                disabled={disabled}
                onClick={() => onPick(p.code)}
                className="flex w-full items-center gap-2.5 px-3 py-2 text-left hover:bg-surface-2 disabled:cursor-not-allowed disabled:opacity-45"
              >
                <PlayerPhoto code={p.code} name={p.name} size={30} />
                <span className="min-w-0 flex-1">
                  <span className="flex items-center gap-1 text-sm font-medium">
                    <span className="truncate">{p.name}</span>
                    <StatusPill player={p} />
                  </span>
                  <span className="flex items-center gap-1 text-[11px] text-muted">
                    <ClubBadge club={club} size={11} />
                    {abbr(clubs, p.team_key)} · {money(p.price)} · {p.ownership.toFixed(1)}%
                    {!clubOk ? " · 3 from club" : !affordable ? " · too expensive" : ""}
                  </span>
                </span>
                <span className="hidden w-24 grid-cols-3 gap-0.5 sm:grid">
                  {gameweeks.slice(0, 3).map((gw) => (
                    <FixtureCell key={gw} club={club} clubs={clubs} gameweek={gw} lens={lensFor(p.position)} compact />
                  ))}
                </span>
                <span className="w-12 text-right">
                  <span className="num block text-sm font-semibold">{sum(p).toFixed(1)}</span>
                  <span className={`num block text-[11px] ${gain >= 0 ? "text-accent-text" : "text-danger"}`}>{signed(gain)}</span>
                </span>
              </button>
            </li>
          );
        })}
        {candidates.length === 0 ? <li className="px-3 py-6 text-center text-sm text-muted">No players match.</li> : null}
      </ul>
    </div>
  );
}

function Planner({
  plan,
  planning,
  error,
  baseline,
  byCode,
  onSuggest,
  onApply,
}: {
  plan: Plan | null;
  planning: boolean;
  error: string | null;
  baseline: Plan | null;
  byCode: Map<number, Player>;
  onSuggest: (wildcard?: boolean) => void;
  onApply: () => void;
}) {
  const gain = plan && baseline ? plan.total_xp - baseline.total_xp : null;
  return (
    <div className="card p-4">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h2 className="text-sm font-semibold">Transfer planner</h2>
          <p className="mt-0.5 text-xs text-muted">
            Optimal transfers, lineups and captains over the horizon, hits included.
          </p>
        </div>
      </div>
      <div className="mt-3 flex gap-2">
        <button
          onClick={() => onSuggest(false)}
          disabled={planning}
          className="flex-1 rounded-lg bg-text px-3 py-2 text-sm font-medium text-bg disabled:opacity-60"
        >
          {planning ? "Solving… (~20s)" : "Suggest transfers"}
        </button>
        <button
          onClick={() => onSuggest(true)}
          disabled={planning}
          className="rounded-lg border border-border px-3 py-2 text-sm disabled:opacity-60"
          title="Plan with a wildcard this gameweek"
        >
          Wildcard
        </button>
      </div>
      {error ? <p className="mt-2 text-xs text-danger">{error}</p> : null}
      {plan ? (
        <div className="mt-4 space-y-3">
          <div className="flex items-baseline justify-between">
            <span className="text-xs text-muted">Plan total (after hits)</span>
            <span className="num font-semibold">
              {plan.total_xp.toFixed(1)}
              {gain != null ? (
                <span className={`ml-1.5 text-xs ${gain >= 0 ? "text-accent-text" : "text-danger"}`}>{signed(gain)}</span>
              ) : null}
            </span>
          </div>
          <ol className="space-y-2">
            {plan.gameweeks.map((g) => (
              <li key={g.gameweek} className="rounded-lg bg-surface-2 px-3 py-2 text-sm">
                <div className="flex items-center justify-between">
                  <span className="label">GW{g.gameweek}</span>
                  <span className="num text-xs text-muted">
                    {g.xp.toFixed(1)} xP{g.hits ? ` · −${g.hits * HIT}` : ""} · C {byCode.get(g.captain)?.name}
                  </span>
                </div>
                {g.transfers_in.length ? (
                  <div className="mt-1 space-y-0.5">
                    {pairTransfers(g.transfers_out, g.transfers_in, byCode).map((s) => (
                      <div key={s.in} className="flex gap-1.5 text-xs">
                        <span className="text-muted line-through">{byCode.get(s.out)?.name}</span>
                        <span className="text-faint">→</span>
                        <span className="font-medium">{byCode.get(s.in)?.name}</span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="mt-1 text-xs text-faint">Roll the transfer</div>
                )}
              </li>
            ))}
          </ol>
          {plan.gameweeks[0]?.transfers_in.length ? (
            <button onClick={onApply} className="w-full rounded-lg bg-accent px-3 py-2 text-sm font-medium text-black">
              Try GW{plan.gameweeks[0].gameweek} moves in the sandbox
            </button>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}

function Captaincy({
  squad,
  byCode,
  gwIndex,
  gameweek,
}: {
  squad: number[];
  byCode: Map<number, Player>;
  gwIndex: number;
  gameweek: number;
}) {
  const options = squad
    .map((c) => byCode.get(c))
    .filter((p): p is Player => !!p)
    .sort((a, b) => (b.xp[gwIndex] ?? 0) - (a.xp[gwIndex] ?? 0))
    .slice(0, 5);
  const top = options[0]?.xp[gwIndex] ?? 1;
  return (
    <div className="card p-4">
      <h2 className="text-sm font-semibold">Captain options · GW{gameweek}</h2>
      <ul className="mt-3 space-y-2">
        {options.map((p) => (
          <li key={p.code} className="flex items-center gap-2.5 text-sm">
            <PlayerPhoto code={p.code} name={p.name} size={26} />
            <span className="w-24 truncate">{p.name}</span>
            <span className="h-1.5 flex-1 overflow-hidden rounded-full bg-surface-2">
              <span className="block h-full rounded-full bg-accent" style={{ width: `${((p.xp[gwIndex] ?? 0) / top) * 100}%` }} />
            </span>
            <span className="num w-9 text-right font-semibold">{(p.xp[gwIndex] ?? 0).toFixed(1)}</span>
            <span className="num w-10 text-right text-xs text-muted" title="Chance of 10+ points (next GW)">
              {gwIndex === 0 ? pct(p.p_haul) : ""}
            </span>
          </li>
        ))}
      </ul>
      <p className="mt-3 text-[11px] text-faint">
        Our experiments found the highest-xP captain maximises season points; upside rules didn&apos;t help ({compact(1500)} path-gameweeks tested).
      </p>
    </div>
  );
}
