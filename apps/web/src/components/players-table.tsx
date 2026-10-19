"use client";

import Link from "next/link";
import { useMemo, useState } from "react";

import { compareHref, MAX_COMPARE } from "@/components/compare";
import { ClubBadge, FixtureCell, PlayerPhoto, PriceMove, StatusPill } from "@/components/ui";
import { abbr, clubMap, compact, lensFor, money, pct, POSITIONS } from "@/lib/format";
import type { Club, Player, Position } from "@/lib/types";

type SortKey =
  | "xp_next"
  | "xp_total"
  | "value"
  | "price"
  | "ownership"
  | "form"
  | "points"
  | "p_haul"
  | "xgi"
  | "move";

const COLUMNS: { key: SortKey; label: string; title: string }[] = [
  { key: "price", label: "Price", title: "Current price" },
  { key: "ownership", label: "Own", title: "Selected by % of managers" },
  { key: "form", label: "Form", title: "Points per gameweek, last 4 gameweeks" },
  { key: "points", label: "Pts", title: "Season points" },
  { key: "xgi", label: "xGI", title: "Season expected goals + assists" },
  { key: "p_haul", label: "Haul", title: "Chance of 10+ points next gameweek" },
  { key: "xp_next", label: "xP", title: "Expected points next gameweek" },
  { key: "xp_total", label: "xP Σ", title: "Expected points over the horizon" },
  { key: "value", label: "xP/£", title: "Horizon expected points per £1m" },
  { key: "move", label: "Price", title: "Chance of a price change before the next deadline" },
];

function sortValue(p: Player, key: SortKey): number {
  switch (key) {
    case "xp_next":
      return p.xp[0] ?? 0;
    case "xp_total":
      return p.xp_total;
    case "value":
      return p.xp_total / p.price;
    case "price":
      return p.price;
    case "ownership":
      return p.ownership;
    case "form":
      return p.season.form;
    case "points":
      return p.season.points;
    case "p_haul":
      return p.p_haul ?? 0;
    case "xgi":
      return p.season.xg + p.season.xa;
    case "move":
      return (p.p_rise ?? 0) - (p.p_fall ?? 0);
  }
}

const PAGE = 60;

export function PlayersTable({ players, clubs, horizon }: { players: Player[]; clubs: Club[]; horizon: number[] }) {
  const byKey = useMemo(() => clubMap(clubs), [clubs]);
  const maxPrice = useMemo(() => Math.ceil(Math.max(...players.map((p) => p.price))), [players]);
  const [position, setPosition] = useState<Position | "ALL">("ALL");
  const [club, setClub] = useState("ALL");
  const [search, setSearch] = useState("");
  const [priceCap, setPriceCap] = useState(maxPrice);
  const [availableOnly, setAvailableOnly] = useState(false);
  const [sort, setSort] = useState<{ key: SortKey; desc: boolean }>({ key: "xp_total", desc: true });
  const [limit, setLimit] = useState(PAGE);
  const [picked, setPicked] = useState<number[]>([]);
  const togglePick = (code: number) =>
    setPicked((current) =>
      current.includes(code)
        ? current.filter((c) => c !== code)
        : current.length < MAX_COMPARE
          ? [...current, code]
          : current,
    );

  const rows = useMemo(() => {
    const q = search.trim().toLowerCase();
    return players
      .filter((p) => position === "ALL" || p.position === position)
      .filter((p) => club === "ALL" || p.team_key === club)
      .filter((p) => p.price <= priceCap)
      .filter((p) => !availableOnly || p.status === "a")
      .filter((p) => !q || p.name.toLowerCase().includes(q) || byKey[p.team_key]?.name.toLowerCase().includes(q))
      .sort((a, b) => (sortValue(b, sort.key) - sortValue(a, sort.key)) * (sort.desc ? 1 : -1));
  }, [players, position, club, priceCap, availableOnly, search, sort, byKey]);

  const fixtureWeeks = horizon.slice(0, 5);
  const toggleSort = (key: SortKey) =>
    setSort((s) => (s.key === key ? { key, desc: !s.desc } : { key, desc: true }));

  return (
    <div className="card overflow-hidden">
      <div className="flex flex-wrap items-center gap-2 border-b border-border p-3">
        <div className="flex rounded-lg bg-surface-2 p-0.5 text-sm">
          {(["ALL", ...POSITIONS] as const).map((pos) => (
            <button
              key={pos}
              onClick={() => setPosition(pos)}
              className={`rounded-md px-2.5 py-1 ${position === pos ? "bg-surface font-medium shadow-sm" : "text-muted hover:text-text"}`}
            >
              {pos === "ALL" ? "All" : pos}
            </button>
          ))}
        </div>
        <select
          value={club}
          onChange={(e) => setClub(e.target.value)}
          className="rounded-lg border border-border bg-surface px-2 py-1.5 text-sm"
          aria-label="Club"
        >
          <option value="ALL">All clubs</option>
          {clubs.map((c) => (
            <option key={c.key} value={c.key}>
              {c.name}
            </option>
          ))}
        </select>
        <label className="flex items-center gap-2 text-sm text-muted">
          <span className="whitespace-nowrap">≤ {money(priceCap)}</span>
          <input
            type="range"
            min={3.5}
            max={maxPrice}
            step={0.5}
            value={priceCap}
            onChange={(e) => setPriceCap(Number(e.target.value))}
            className="w-28 accent-[var(--accent)]"
          />
        </label>
        <label className="flex items-center gap-1.5 text-sm text-muted">
          <input
            type="checkbox"
            checked={availableOnly}
            onChange={(e) => setAvailableOnly(e.target.checked)}
            className="accent-[var(--accent)]"
          />
          Available only
        </label>
        <input
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search player or club"
          className="ml-auto w-full rounded-lg border border-border bg-surface px-3 py-1.5 text-sm outline-none focus:border-accent sm:w-56"
        />
      </div>

      <div className="scroll-thin overflow-x-auto">
        <table className="w-full min-w-[1100px] text-sm">
          <thead>
            <tr className="border-b border-border text-left">
              <th className="label sticky left-0 z-10 bg-surface py-2 pr-2 pl-3 font-medium">
                <span className="flex items-center gap-2.5">
                  <span className="w-4" title={`Tick up to ${MAX_COMPARE} players to compare`}>
                    ⇄
                  </span>
                  Player
                </span>
              </th>
              {COLUMNS.map((c) => (
                <th key={c.key} className="py-2 pr-3 text-right" title={c.title}>
                  <button
                    onClick={() => toggleSort(c.key)}
                    className={`label inline-flex items-center gap-0.5 font-medium hover:text-text ${sort.key === c.key ? "text-text" : ""}`}
                  >
                    {c.label}
                    <span className="w-2 text-[9px]">{sort.key === c.key ? (sort.desc ? "▼" : "▲") : ""}</span>
                  </button>
                </th>
              ))}
              <th className="label py-2 pr-3 pl-2 font-medium" title="Fixtures, coloured by the model's attacking or clean-sheet outlook for the player's position">
                Next {fixtureWeeks.length}
              </th>
            </tr>
          </thead>
          <tbody>
            {rows.slice(0, limit).map((p) => {
              const club = byKey[p.team_key];
              return (
                <tr key={p.code} className="group border-b border-border/60 last:border-0 hover:bg-surface-2/60">
                  <td className="sticky left-0 z-10 bg-surface py-1.5 pr-2 pl-3 group-hover:bg-surface-2">
                    <span className="flex items-center gap-2.5">
                    <input
                      type="checkbox"
                      checked={picked.includes(p.code)}
                      onChange={() => togglePick(p.code)}
                      disabled={!picked.includes(p.code) && picked.length >= MAX_COMPARE}
                      aria-label={`Compare ${p.name}`}
                      className="h-4 w-4 shrink-0 accent-[var(--accent)]"
                    />
                    <Link href={`/players/${p.code}`} className="flex items-center gap-2.5">
                      <PlayerPhoto code={p.code} name={p.name} size={32} />
                      <span className="min-w-0">
                        <span className="flex items-center gap-1.5">
                          <span className="truncate font-medium group-hover:text-accent-text">{p.name}</span>
                          <StatusPill player={p} />
                        </span>
                        <span className="flex items-center gap-1 text-xs text-muted">
                          <ClubBadge club={club} size={12} />
                          {abbr(byKey, p.team_key)} · {p.position}
                        </span>
                      </span>
                    </Link>
                    </span>
                  </td>
                  <td className="num py-1.5 pr-3 text-right">{p.price.toFixed(1)}</td>
                  <td className="num py-1.5 pr-3 text-right text-muted">{p.ownership.toFixed(1)}%</td>
                  <td className="num py-1.5 pr-3 text-right">{p.season.form.toFixed(1)}</td>
                  <td className="num py-1.5 pr-3 text-right">{p.season.points}</td>
                  <td className="num py-1.5 pr-3 text-right text-muted">{(p.season.xg + p.season.xa).toFixed(1)}</td>
                  <td className="num py-1.5 pr-3 text-right text-muted">{pct(p.p_haul)}</td>
                  <td className="num py-1.5 pr-3 text-right font-semibold">{(p.xp[0] ?? 0).toFixed(1)}</td>
                  <td className="num py-1.5 pr-3 text-right">
                    <span className="rounded bg-accent-soft px-1.5 py-0.5 font-semibold text-accent-text">
                      {p.xp_total.toFixed(1)}
                    </span>
                  </td>
                  <td className="num py-1.5 pr-3 text-right text-muted">{(p.xp_total / p.price).toFixed(2)}</td>
                  <td className="num py-1.5 pr-3 text-right text-xs">
                    <PriceMove player={p} />
                    {p.net_transfers != null && Math.abs(p.net_transfers) >= 1000 ? (
                      <div className="text-[10px] text-faint">{p.net_transfers > 0 ? "+" : ""}{compact(p.net_transfers)}</div>
                    ) : null}
                  </td>
                  <td className="py-1.5 pr-3 pl-2">
                    <div className="grid w-[260px] grid-cols-5 gap-1">
                      {fixtureWeeks.map((gw) => (
                        <FixtureCell key={gw} club={club} clubs={byKey} gameweek={gw} lens={lensFor(p.position)} compact />
                      ))}
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="flex items-center justify-between border-t border-border px-3 py-2 text-xs text-muted">
        <span>
          {rows.length} players · horizon GW{horizon[0]}–{horizon[horizon.length - 1]}
        </span>
        {picked.length ? (
          <span className="flex items-center gap-3">
            <button onClick={() => setPicked([])} className="hover:text-text">
              Clear
            </button>
            <Link
              href={compareHref(picked)}
              className="rounded-md bg-accent px-2.5 py-1 font-medium text-black"
            >
              Compare {picked.length}
            </Link>
          </span>
        ) : null}
        {rows.length > limit ? (
          <button onClick={() => setLimit((l) => l + PAGE)} className="font-medium text-accent-text hover:underline">
            Show more
          </button>
        ) : null}
      </div>
    </div>
  );
}
