import type { Metadata } from "next";
import Image from "next/image";
import Link from "next/link";
import { Suspense } from "react";

import { ClubBadge, FixtureCell, Skeleton } from "@/components/ui";
import { api } from "@/lib/api";
import { badgeUrl, clubMap, pct } from "@/lib/format";
import type { Club } from "@/lib/types";

export const metadata: Metadata = { title: "Clubs" };

/** Attack (goals scored vs average) against defence (goals conceded vs average). */
function StrengthMap({ clubs }: { clubs: Club[] }) {
  const xs = clubs.map((c) => c.attack);
  const ys = clubs.map((c) => c.defence);
  const [x0, x1] = [Math.min(...xs) * 0.95, Math.max(...xs) * 1.05];
  const [y0, y1] = [Math.min(...ys) * 0.95, Math.max(...ys) * 1.05];
  const left = (x: number) => ((x - x0) / (x1 - x0)) * 100;
  // Fewer goals conceded is better: put strong defences at the top.
  const top = (y: number) => ((y - y0) / (y1 - y0)) * 100;
  return (
    <div className="card p-5">
      <div className="mb-2 flex items-baseline justify-between">
        <h2 className="text-sm font-semibold">Strength map</h2>
        <span className="text-xs text-muted">time-weighted ratings on goals and xG · 1.0 = league average</span>
      </div>
      <div className="relative mx-auto aspect-[16/7] max-h-[440px] w-full">
        <div className="absolute inset-0 rounded-lg bg-surface-2/50" />
        <div className="absolute top-0 bottom-0 border-l border-dashed border-border" style={{ left: `${left(1)}%` }} />
        <div className="absolute right-0 left-0 border-t border-dashed border-border" style={{ top: `${top(1)}%` }} />
        <span className="label absolute top-2 right-3">strong attack · strong defence</span>
        <span className="label absolute bottom-2 left-3">weak attack · weak defence</span>
        {clubs.map((c) => (
          <Link
            key={c.key}
            href={`/clubs/${c.key}`}
            className="group absolute -translate-x-1/2 -translate-y-1/2"
            style={{ left: `${left(c.attack)}%`, top: `${top(c.defence)}%` }}
            title={`${c.name}: attack ×${c.attack.toFixed(2)}, concedes ×${c.defence.toFixed(2)}`}
          >
            <Image src={badgeUrl(c.fpl_team_code)} alt={c.name} width={30} height={30} className="transition group-hover:scale-125" />
          </Link>
        ))}
      </div>
      <div className="mt-2 flex justify-between text-xs text-muted">
        <span>← scores fewer</span>
        <span>attack →</span>
      </div>
    </div>
  );
}

async function Clubs() {
  const [clubs, meta] = await Promise.all([api.clubs(), api.meta()]);
  const byKey = clubMap(clubs);
  const weeks = meta.horizon;
  const sorted = [...clubs].sort((a, b) => b.attack / b.defence - a.attack / a.defence);
  return (
    <div className="space-y-6">
      <StrengthMap clubs={clubs} />
      <div className="card overflow-hidden">
        <div className="scroll-thin overflow-x-auto">
          <table className="w-full min-w-[900px] text-sm">
            <thead>
              <tr className="border-b border-border text-right">
                <th className="label py-2 pl-4 text-left font-medium">Club</th>
                <th className="label py-2 pl-3 font-medium">W-D-L</th>
                <th className="label py-2 pl-3 font-medium">GF–GA</th>
                <th className="label py-2 pl-3 font-medium">xG–xGA</th>
                <th className="label py-2 pl-3 font-medium" title="Goals scored vs league average">Attack</th>
                <th className="label py-2 pl-3 font-medium" title="Goals conceded vs league average (lower is better)">Defence</th>
                <th className="label py-2 pl-3 font-medium">Next CS</th>
                <th className="label py-2 pr-4 pl-3 text-left font-medium">Next {weeks.length} (attack)</th>
              </tr>
            </thead>
            <tbody>
              {sorted.map((c) => (
                <tr key={c.key} className="border-b border-border/60 text-right last:border-0 hover:bg-surface-2/60">
                  <td className="py-2 pl-4 text-left">
                    <Link href={`/clubs/${c.key}`} className="flex items-center gap-2 font-medium hover:text-accent-text">
                      <ClubBadge club={c} size={20} />
                      {c.name}
                    </Link>
                  </td>
                  <td className="num py-2 pl-3">
                    {c.record.won}-{c.record.drawn}-{c.record.lost}
                  </td>
                  <td className="num py-2 pl-3 text-muted">
                    {c.record.goals_for}–{c.record.goals_against}
                  </td>
                  <td className="num py-2 pl-3 text-muted">
                    {c.record.xg_for.toFixed(1)}–{c.record.xg_against.toFixed(1)}
                  </td>
                  <td className={`num py-2 pl-3 font-medium ${c.attack >= 1.1 ? "text-accent-text" : c.attack <= 0.85 ? "text-danger" : ""}`}>
                    ×{c.attack.toFixed(2)}
                  </td>
                  <td className={`num py-2 pl-3 font-medium ${c.defence <= 0.85 ? "text-accent-text" : c.defence >= 1.15 ? "text-danger" : ""}`}>
                    ×{c.defence.toFixed(2)}
                  </td>
                  <td className="num py-2 pl-3">{pct(c.fixtures.find((f) => f.gameweek === weeks[0])?.clean_sheet)}</td>
                  <td className="py-2 pr-4 pl-3">
                    <div className="grid w-[300px] grid-cols-5 gap-1">
                      {weeks.map((gw) => (
                        <FixtureCell key={gw} club={c} clubs={byKey} gameweek={gw} lens="attack" compact />
                      ))}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

export default function Page() {
  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-xl font-semibold tracking-tight">Clubs</h1>
        <p className="mt-1 text-sm text-muted">Attack and defence strength, form against expected goals, and what&apos;s coming up.</p>
      </div>
      <Suspense fallback={<Skeleton className="h-[900px]" />}>
        <Clubs />
      </Suspense>
    </div>
  );
}
