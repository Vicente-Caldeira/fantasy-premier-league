import Link from "next/link";
import { notFound } from "next/navigation";
import { Suspense } from "react";

import { ClubBadge, OutcomeBar, PlayerPhoto, PriceMove, SectionTitle, Skeleton, Stat, StatusPill } from "@/components/ui";
import { api, ApiError } from "@/lib/api";
import { clubMap, money, pct } from "@/lib/format";
import type { Player } from "@/lib/types";

function rowFactory() {
  return function row(p: Player) {
    return (
                <tr key={p.code} className="border-b border-border/60 text-right last:border-0 hover:bg-surface-2/60">
                  <td className="py-1.5 pl-5 text-left">
                    <Link href={`/players/${p.code}`} className="flex items-center gap-2 hover:text-accent-text">
                      <PlayerPhoto code={p.code} name={p.name} size={28} />
                      <span className="font-medium">{p.name}</span>
                      <span className="text-xs text-muted">{p.position}</span>
                      <StatusPill player={p} />
                    </Link>
                  </td>
                  <td className="num py-1.5 pl-3">{money(p.price)}</td>
                  <td className="num py-1.5 pl-3 text-muted">{p.ownership.toFixed(1)}%</td>
                  <td className="num py-1.5 pl-3">{p.season.points}</td>
                  {p.xp.map((x, i) => (
                    <td key={i} className="num py-1.5 pl-3 text-muted">
                      {x.toFixed(1)}
                    </td>
                  ))}
                  <td className="num py-1.5 pl-3 font-semibold">{p.xp_total.toFixed(1)}</td>
                  <td className="num py-1.5 pr-5 pl-3 text-xs">
                    <PriceMove player={p} />
                  </td>
                </tr>
    );
  };
}

async function ClubPage({ params }: { params: Promise<{ key: string }> }) {
  const { key } = await params;
  let detail;
  try {
    detail = await api.club(key);
  } catch (error) {
    if (error instanceof ApiError && error.status === 404) notFound();
    throw error;
  }
  const [clubs, meta] = await Promise.all([api.clubs(), api.meta()]);
  const byKey = clubMap(clubs);
  const { club, results, players } = detail;
  const r = club.record;
  const upcoming = club.fixtures.filter((f) => meta.horizon.includes(f.gameweek) || f.gameweek >= (meta.next_gameweek ?? 0)).slice(0, 8);
  const sortedSquad = [...players].sort((a, b) => b.xp_total - a.xp_total);
  const row = rowFactory();
  const squad = sortedSquad.filter((p) => p.xp_total >= 1);
  const fringe = sortedSquad.filter((p) => p.xp_total < 1);
  const attackRank = [...clubs].sort((a, b) => b.attack - a.attack).findIndex((c) => c.key === club.key) + 1;
  const defenceRank = [...clubs].sort((a, b) => a.defence - b.defence).findIndex((c) => c.key === club.key) + 1;

  return (
    <div className="space-y-6">
      <div className="card flex flex-wrap items-center gap-6 p-5">
        <ClubBadge club={club} size={72} />
        <div className="min-w-0 flex-1">
          <h1 className="text-2xl font-semibold tracking-tight">{club.name}</h1>
          <p className="mt-1 text-sm text-muted">
            {r.won}W {r.drawn}D {r.lost}L · goals {r.goals_for}–{r.goals_against} · xG {r.xg_for.toFixed(1)}–{r.xg_against.toFixed(1)}
          </p>
        </div>
        <div className="grid grid-cols-2 gap-x-8 gap-y-3 sm:grid-cols-4">
          <Stat label="Attack" value={`×${club.attack.toFixed(2)}`} hint={`#${attackRank} in the league`} emphasis />
          <Stat label="Defence" value={`×${club.defence.toFixed(2)}`} hint={`#${defenceRank} · goals conceded`} emphasis />
          <Stat label="Next CS" value={pct(upcoming[0]?.clean_sheet)} hint={upcoming[0] ? `GW${upcoming[0].gameweek}` : undefined} />
          <Stat
            label="Points / game"
            value={r.played ? ((3 * r.won + r.drawn) / r.played).toFixed(2) : "–"}
            hint={`${r.played} played`}
          />
        </div>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <section className="card p-5">
          <SectionTitle aside="win / draw / loss · expected goals · clean sheet">Upcoming</SectionTitle>
          <ul className="space-y-3">
            {upcoming.map((f) => {
              const opp = byKey[f.opponent];
              return (
                <li key={`${f.gameweek}-${f.opponent}`} className="grid grid-cols-[3rem_1fr_7rem_3.5rem] items-center gap-3 text-sm">
                  <span className="num text-muted">GW{f.gameweek}</span>
                  <Link href={`/clubs/${f.opponent}`} className="flex items-center gap-2 truncate hover:text-accent-text">
                    <ClubBadge club={opp} size={18} />
                    <span className="truncate">{opp?.name}</span>
                    <span className="text-faint">({f.is_home ? "H" : "A"})</span>
                  </Link>
                  <span>
                    <OutcomeBar win={f.win} draw={f.draw} loss={f.loss} />
                    <span className="num mt-1 block text-[11px] text-muted">
                      {f.goals_for.toFixed(1)}–{f.goals_against.toFixed(1)} xG
                    </span>
                  </span>
                  <span className="num text-right font-medium">{pct(f.clean_sheet)}</span>
                </li>
              );
            })}
          </ul>
        </section>

        <section className="card p-5">
          <SectionTitle aside="goals (xG)">Results</SectionTitle>
          <ul className="divide-y divide-border/60">
            {[...results].reverse().map((m) => {
              const home = m.home === club.key;
              const opp = byKey[home ? m.away : m.home];
              const gf = home ? m.home_goals : m.away_goals;
              const ga = home ? m.away_goals : m.home_goals;
              const xf = home ? m.home_xg : m.away_xg;
              const xa = home ? m.away_xg : m.home_xg;
              const outcome = gf! > ga! ? "W" : gf === ga ? "D" : "L";
              return (
                <li key={m.match_key} className="flex items-center gap-3 py-2 text-sm">
                  <span
                    className={`grid h-6 w-6 place-items-center rounded text-[11px] font-bold ${
                      outcome === "W" ? "bg-accent text-black" : outcome === "D" ? "bg-surface-2" : "bg-danger/80 text-white"
                    }`}
                  >
                    {outcome}
                  </span>
                  <span className="num w-10 text-muted">GW{m.gameweek}</span>
                  <ClubBadge club={opp} size={18} />
                  <span className="flex-1 truncate">
                    {opp?.name} <span className="text-faint">({home ? "H" : "A"})</span>
                  </span>
                  <span className="num font-semibold">
                    {gf}–{ga}
                  </span>
                  <span className="num w-20 text-right text-xs text-muted">
                    ({xf?.toFixed(1) ?? "–"}–{xa?.toFixed(1) ?? "–"})
                  </span>
                </li>
              );
            })}
          </ul>
        </section>
      </div>

      <section className="card overflow-hidden">
        <div className="p-5 pb-0">
          <SectionTitle aside={`expected points GW${meta.horizon[0]}–${meta.horizon[meta.horizon.length - 1]}`}>Squad</SectionTitle>
        </div>
        <div className="scroll-thin overflow-x-auto">
          <table className="w-full min-w-[720px] text-sm">
            <thead>
              <tr className="border-b border-border text-right">
                <th className="label py-2 pl-5 text-left font-medium">Player</th>
                <th className="label py-2 pl-3 font-medium">Price</th>
                <th className="label py-2 pl-3 font-medium">Own</th>
                <th className="label py-2 pl-3 font-medium">Pts</th>
                {meta.horizon.map((gw) => (
                  <th key={gw} className="label py-2 pl-3 font-medium">
                    GW{gw}
                  </th>
                ))}
                <th className="label py-2 pl-3 font-medium">Σ</th>
                <th className="label py-2 pr-5 pl-3 font-medium">Price</th>
              </tr>
            </thead>
            <tbody>
              {squad.map(row)}
            </tbody>
          </table>
        </div>
        {fringe.length ? (
          <details className="border-t border-border">
            <summary className="cursor-pointer px-5 py-2.5 text-xs text-muted hover:text-text">
              {fringe.length} fringe or unavailable players
            </summary>
            <div className="scroll-thin overflow-x-auto">
              <table className="w-full min-w-[720px] text-sm">
                <tbody>{fringe.map(row)}</tbody>
              </table>
            </div>
          </details>
        ) : null}
      </section>
    </div>
  );
}

export default function Page(props: PageProps<"/clubs/[key]">) {
  return (
    <Suspense fallback={<Skeleton className="h-[800px]" />}>
      <ClubPage params={props.params} />
    </Suspense>
  );
}
