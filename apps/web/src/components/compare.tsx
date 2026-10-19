"use client";

import Link from "next/link";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useMemo, useState } from "react";

import { PlayerPhoto } from "@/components/photo";
import { ClubBadge, FixtureCell, PriceMove, StatusPill } from "@/components/ui";
import { abbr, clubMap, lensFor, money, pct } from "@/lib/format";
import type { Club, Player } from "@/lib/types";

export const MAX_COMPARE = 4;
const COLORS = ["var(--accent)", "oklch(0.7 0.15 250)", "oklch(0.75 0.15 70)", "oklch(0.68 0.18 330)"];

export const compareHref = (codes: number[]) => `/compare?p=${codes.join(",")}`;

export function parseCodes(value: string | null): number[] {
  return (value ?? "")
    .split(",")
    .map(Number)
    .filter((n) => Number.isInteger(n) && n > 0)
    .slice(0, MAX_COMPARE);
}

const per90 = (value: number, minutes: number) => (minutes >= 90 ? (value / minutes) * 90 : null);

type Row = {
  label: string;
  title?: string;
  value: (p: Player) => number | null;
  format: (v: number) => string;
  best?: "high" | "low";
};

function XpChart({ players, horizon }: { players: Player[]; horizon: number[] }) {
  // Zoom to the data (with a little headroom) so differences of half a point are visible.
  const all = players.flatMap((p) => p.xp);
  const lo = Math.max(0, Math.floor(Math.min(...all) - 0.5));
  const hi = Math.ceil(Math.max(...all) + 0.5);
  const w = 100;
  const h = 40;
  const x = (i: number) => (horizon.length === 1 ? w / 2 : (i / (horizon.length - 1)) * (w - 8) + 4);
  const y = (v: number) => h - ((v - lo) / (hi - lo)) * (h - 4) - 2;
  return (
    <div>
      <div className="relative">
      <span className="num absolute top-0 left-0 text-[10px] text-faint">{hi}</span>
      <span className="num absolute bottom-0 left-0 text-[10px] text-faint">{lo}</span>
      <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" className="h-36 w-full">
        {[0.25, 0.5, 0.75].map((f) => (
          <line key={f} x1={0} x2={w} y1={h * f} y2={h * f} stroke="var(--border)" strokeWidth={0.3} vectorEffect="non-scaling-stroke" />
        ))}
        {players.map((p, i) => (
          <polyline
            key={p.code}
            fill="none"
            stroke={COLORS[i]}
            strokeWidth={2}
            vectorEffect="non-scaling-stroke"
            strokeLinejoin="round"
            points={p.xp.map((v, j) => `${x(j)},${y(v)}`).join(" ")}
          />
        ))}
      </svg>
      </div>
      <div className="mt-1 flex justify-between px-[4%] text-[10px] text-faint">
        {horizon.map((gw) => (
          <span key={gw} className="num">
            GW{gw}
          </span>
        ))}
      </div>
    </div>
  );
}

export function Compare({ players, clubs, horizon }: { players: Player[]; clubs: Club[]; horizon: number[] }) {
  const router = useRouter();
  const pathname = usePathname();
  const params = useSearchParams();
  const byCode = useMemo(() => new Map(players.map((p) => [p.code, p])), [players]);
  const byKey = useMemo(() => clubMap(clubs), [clubs]);
  const codes = parseCodes(params.get("p")).filter((c) => byCode.has(c));
  const chosen = codes.map((c) => byCode.get(c)!);
  const [search, setSearch] = useState("");

  const setCodes = (next: number[]) =>
    router.replace(next.length ? `${pathname}?p=${next.join(",")}` : pathname, { scroll: false });

  const q = search.trim().toLowerCase();
  const matches = q
    ? players
        .filter((p) => !codes.includes(p.code))
        .filter((p) => p.name.toLowerCase().includes(q) || byKey[p.team_key]?.name.toLowerCase().includes(q))
        .sort((a, b) => b.xp_total - a.xp_total)
        .slice(0, 8)
    : [];

  const groups: { title: string; rows: Row[] }[] = [
    {
      title: "Outlook",
      rows: [
        ...horizon.map<Row>((gw, i) => ({ label: `xP GW${gw}`, value: (p) => p.xp[i] ?? 0, format: (v) => v.toFixed(1), best: "high" })),
        { label: `xP next ${horizon.length}`, value: (p) => p.xp_total, format: (v) => v.toFixed(1), best: "high" },
        { label: "xP per £m", value: (p) => p.xp_total / p.price, format: (v) => v.toFixed(2), best: "high" },
        { label: "Haul chance", title: "10+ points next gameweek", value: (p) => p.p_haul, format: (v) => pct(v), best: "high" },
        { label: "Plays 60+", title: "next gameweek", value: (p) => p.p_60, format: (v) => pct(v), best: "high" },
        { label: "80% range", title: "next gameweek", value: (p) => p.p90, format: (v) => v.toFixed(0), best: "high" },
      ],
    },
    {
      title: "Season",
      rows: [
        { label: "Points", value: (p) => p.season.points, format: (v) => v.toFixed(0), best: "high" },
        { label: "Form", title: "points per GW, last 4", value: (p) => p.season.form, format: (v) => v.toFixed(1), best: "high" },
        { label: "Minutes", value: (p) => p.season.minutes, format: (v) => v.toFixed(0), best: "high" },
        { label: "Goals", value: (p) => p.season.goals, format: (v) => v.toFixed(0), best: "high" },
        { label: "Assists", value: (p) => p.season.assists, format: (v) => v.toFixed(0), best: "high" },
        { label: "xG / 90", value: (p) => per90(p.season.xg, p.season.minutes), format: (v) => v.toFixed(2), best: "high" },
        { label: "xA / 90", value: (p) => per90(p.season.xa, p.season.minutes), format: (v) => v.toFixed(2), best: "high" },
        { label: "xGI / 90", value: (p) => per90(p.season.xg + p.season.xa, p.season.minutes), format: (v) => v.toFixed(2), best: "high" },
        { label: "Points / 90", value: (p) => per90(p.season.points, p.season.minutes), format: (v) => v.toFixed(2), best: "high" },
        { label: "Bonus", value: (p) => p.season.bonus, format: (v) => v.toFixed(0), best: "high" },
        { label: "Clean sheets", value: (p) => p.season.clean_sheets, format: (v) => v.toFixed(0), best: "high" },
      ],
    },
  ];

  const cols = `10rem repeat(${Math.max(chosen.length, 1)}, minmax(0, 1fr))`;

  return (
    <div className="space-y-4">
      <div className="card p-3">
        <div className="relative">
          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            disabled={codes.length >= MAX_COMPARE}
            placeholder={codes.length >= MAX_COMPARE ? `Up to ${MAX_COMPARE} players` : "Add a player to compare…"}
            className="w-full rounded-lg border border-border bg-surface px-3 py-2 text-sm outline-none focus:border-accent disabled:opacity-60"
          />
          {matches.length ? (
            <ul className="absolute z-20 mt-1 w-full overflow-hidden rounded-lg border border-border bg-surface shadow-lg">
              {matches.map((p) => (
                <li key={p.code}>
                  <button
                    onClick={() => {
                      setCodes([...codes, p.code]);
                      setSearch("");
                    }}
                    className="flex w-full items-center gap-2.5 px-3 py-2 text-left text-sm hover:bg-surface-2"
                  >
                    <PlayerPhoto code={p.code} name={p.name} size={26} />
                    <span className="font-medium">{p.name}</span>
                    <span className="text-xs text-muted">
                      {abbr(byKey, p.team_key)} · {p.position} · {money(p.price)}
                    </span>
                    <span className="num ml-auto text-xs font-semibold">{p.xp_total.toFixed(1)} xP</span>
                  </button>
                </li>
              ))}
            </ul>
          ) : null}
        </div>
      </div>

      {chosen.length === 0 ? (
        <p className="py-16 text-center text-sm text-muted">
          Pick up to {MAX_COMPARE} players above, or tick players in the{" "}
          <Link href="/players" className="text-accent-text hover:underline">
            players table
          </Link>
          .
        </p>
      ) : (
        <div className="card overflow-x-auto">
          <div className="min-w-[640px]">
            {/* Player headers */}
            <div className="grid border-b border-border" style={{ gridTemplateColumns: cols }}>
              <div />
              {chosen.map((p, i) => {
                const club = byKey[p.team_key];
                return (
                  <div key={p.code} className="relative flex flex-col items-center gap-1.5 px-3 py-4 text-center">
                    <button
                      onClick={() => setCodes(codes.filter((c) => c !== p.code))}
                      aria-label={`Remove ${p.name}`}
                      className="absolute top-2 right-2 text-xs text-faint hover:text-text"
                    >
                      ✕
                    </button>
                    <PlayerPhoto code={p.code} name={p.name} size={56} />
                    <Link href={`/players/${p.code}`} className="flex items-center gap-1.5 font-semibold hover:text-accent-text">
                      <span className="h-2 w-2 rounded-full" style={{ background: COLORS[i] }} />
                      {p.name}
                    </Link>
                    <span className="flex items-center gap-1 text-xs text-muted">
                      <ClubBadge club={club} size={12} />
                      {abbr(byKey, p.team_key)} · {p.position} · {money(p.price)}
                    </span>
                    <span className="flex items-center gap-2 text-xs text-muted">
                      {p.ownership.toFixed(1)}% owned · <PriceMove player={p} />
                    </span>
                    <StatusPill player={p} />
                  </div>
                );
              })}
            </div>

            <div className="border-b border-border px-4 py-3">
              <div className="label mb-1">Expected points by gameweek</div>
              <XpChart players={chosen} horizon={horizon} />
            </div>

            {groups.map((group) => (
              <div key={group.title}>
                <div className="label bg-surface-2/60 px-4 py-1.5">{group.title}</div>
                {group.rows.map((row) => {
                  // Rank on the displayed (rounded) values, so a highlight is never a hidden tie.
                  const values = chosen.map((p) => {
                    const v = row.value(p);
                    return v == null ? null : Number(row.format(v).replace(/[^0-9.-]/g, ""));
                  });
                  const numeric = values.filter((v): v is number => v != null && !Number.isNaN(v));
                  const best =
                    chosen.length > 1 && numeric.length > 1 && row.best
                      ? row.best === "high"
                        ? Math.max(...numeric)
                        : Math.min(...numeric)
                      : null;
                  const unique = best != null && numeric.filter((v) => v === best).length === 1;
                  return (
                    <div
                      key={row.label}
                      className="grid border-b border-border/60 text-sm last:border-0"
                      style={{ gridTemplateColumns: cols }}
                    >
                      <div className="px-4 py-1.5 text-muted" title={row.title}>
                        {row.label}
                      </div>
                      {values.map((v, i) => (
                        <div
                          key={chosen[i].code}
                          className={`num px-3 py-1.5 text-center ${
                            unique && v === best ? "bg-accent-soft font-semibold text-accent-text" : ""
                          }`}
                        >
                          {v == null ? (
                            <span className="text-faint">–</span>
                          ) : row.label === "80% range" ? (
                            `${chosen[i].p10}–${chosen[i].p90}`
                          ) : (
                            row.format(row.value(chosen[i])!)
                          )}
                        </div>
                      ))}
                    </div>
                  );
                })}
              </div>
            ))}

            <div className="label bg-surface-2/60 px-4 py-1.5">Fixtures</div>
            <div className="grid text-sm" style={{ gridTemplateColumns: cols }}>
              <div className="px-4 py-2 text-muted">Next {horizon.length}</div>
              {chosen.map((p) => (
                <div key={p.code} className="grid grid-cols-5 gap-1 px-3 py-2">
                  {horizon.map((gw) => (
                    <FixtureCell key={gw} club={byKey[p.team_key]} clubs={byKey} gameweek={gw} lens={lensFor(p.position)} compact />
                  ))}
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
