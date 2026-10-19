import Link from "next/link";
import { notFound } from "next/navigation";
import { Suspense } from "react";

import { PointsChart } from "@/components/charts";
import { Bar, ClubBadge, FixtureCell, PlayerPhoto, PriceMove, SectionTitle, Skeleton, Stat, StatusPill } from "@/components/ui";
import { api, ApiError } from "@/lib/api";
import { clubMap, lensFor, money, pct, STATUS_LABEL } from "@/lib/format";
import type { MatchRow } from "@/lib/types";

const per90 = (value: number, minutes: number) => (minutes > 0 ? ((value / minutes) * 90).toFixed(2) : "–");

function MatchLog({ matches, clubs }: { matches: MatchRow[]; clubs: Record<string, { short_name: string }> }) {
  return (
    <div className="scroll-thin overflow-x-auto">
      <table className="w-full min-w-[640px] text-sm">
        <thead>
          <tr className="border-b border-border text-right">
            <th className="label py-2 text-left font-medium">GW</th>
            <th className="label py-2 pl-2 text-left font-medium">Opponent</th>
            {["Min", "Pts", "G", "A", "xG", "xA", "CS", "Bonus", "BPS", "Price"].map((h) => (
              <th key={h} className="label py-2 pl-3 font-medium">
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {[...matches].reverse().map((m) => (
            <tr key={`${m.season}-${m.gameweek}-${m.opponent}`} className="border-b border-border/60 text-right last:border-0">
              <td className="num py-1.5 text-left text-muted">{m.gameweek}</td>
              <td className="py-1.5 pl-2 text-left">
                {clubs[m.opponent]?.short_name ?? m.opponent} <span className="text-faint">({m.is_home ? "H" : "A"})</span>
              </td>
              <td className={`num py-1.5 pl-3 ${m.minutes === 0 ? "text-faint" : ""}`}>{m.minutes}</td>
              <td className={`num py-1.5 pl-3 font-semibold ${m.points >= 10 ? "text-accent-text" : ""}`}>{m.points}</td>
              <td className="num py-1.5 pl-3">{m.goals || <span className="text-faint">0</span>}</td>
              <td className="num py-1.5 pl-3">{m.assists || <span className="text-faint">0</span>}</td>
              <td className="num py-1.5 pl-3 text-muted">{m.xg?.toFixed(2) ?? "–"}</td>
              <td className="num py-1.5 pl-3 text-muted">{m.xa?.toFixed(2) ?? "–"}</td>
              <td className="num py-1.5 pl-3">{m.clean_sheet || <span className="text-faint">0</span>}</td>
              <td className="num py-1.5 pl-3">{m.bonus || <span className="text-faint">0</span>}</td>
              <td className="num py-1.5 pl-3 text-muted">{m.bps}</td>
              <td className="num py-1.5 pl-3 text-muted">{m.price.toFixed(1)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

async function Profile({ params }: { params: Promise<{ code: string }> }) {
  const { code } = await params;
  const id = Number(code);
  if (!Number.isInteger(id)) notFound();
  let detail;
  try {
    detail = await api.player(id);
  } catch (error) {
    if (error instanceof ApiError && error.status === 404) notFound();
    throw error;
  }
  const [clubs, meta] = await Promise.all([api.clubs(), api.meta()]);
  const byKey = clubMap(clubs);
  const { player: p, upcoming, matches, past_seasons } = detail;
  const club = byKey[p.team_key];
  const s = p.season;
  const lens = lensFor(p.position);
  const thisSeason = matches.filter((m) => m.season === meta.season);
  const lastSeason = matches.filter((m) => m.season !== meta.season);
  const next = upcoming[0];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="card overflow-hidden">
        <div className="flex flex-wrap items-center gap-5 p-5">
          <PlayerPhoto code={p.code} name={p.name} size={88} />
          <div className="min-w-0 flex-1">
            <div className="flex flex-wrap items-center gap-2">
              <h1 className="text-2xl font-semibold tracking-tight">{p.name}</h1>
              <StatusPill player={p} />
            </div>
            <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-sm text-muted">
              <Link href={`/clubs/${p.team_key}`} className="flex items-center gap-1.5 hover:text-text">
                <ClubBadge club={club} size={16} />
                {club?.name}
              </Link>
              <span>{p.position}</span>
              <span className="num text-text">{money(p.price)}</span>
              <span>{p.ownership.toFixed(1)}% owned</span>
              <span className="flex items-center gap-1">
                Price tonight <PriceMove player={p} />
              </span>
            </div>
          </div>
          <div className="grid grid-cols-2 gap-x-8 gap-y-3 sm:grid-cols-4">
            <Stat
              label={`xP GW${next?.gameweek ?? ""}`}
              value={(p.xp[0] ?? 0).toFixed(1)}
              hint={p.p10 != null ? `80% range ${p.p10}–${p.p90}` : undefined}
              emphasis
            />
            <Stat label={`xP next ${p.xp.length}`} value={p.xp_total.toFixed(1)} hint={`${(p.xp_total / p.price).toFixed(2)} per £m`} emphasis />
            <Stat label="Haul chance" value={pct(p.p_haul)} hint="10+ points next GW" />
            <Stat label="Plays 60+" value={pct(p.p_60)} hint="next gameweek" />
          </div>
        </div>
        {p.status !== "a" ? (
          <div className="border-t border-border bg-warn/10 px-5 py-2.5 text-sm">
            <span className="font-medium">{STATUS_LABEL[p.status]}</span>
            {p.news ? <span className="text-muted"> · {p.news}</span> : null}
          </div>
        ) : null}
      </div>

      <div className="grid gap-6 lg:grid-cols-[1.4fr_1fr]">
        {/* Next gameweeks */}
        <section className="card p-5">
          <SectionTitle aside={lens === "attack" ? "coloured by expected goals for" : "coloured by clean-sheet odds"}>
            Next gameweeks
          </SectionTitle>
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border text-right">
                <th className="label py-2 text-left font-medium">GW</th>
                <th className="label w-28 py-2 pl-2 text-left font-medium">Fixture</th>
                <th className="label py-2 pl-3 text-left font-medium">xP</th>
                <th className="label py-2 pl-3 font-medium">60+</th>
                <th className="label py-2 pl-3 font-medium">xG</th>
                <th className="label py-2 pl-3 font-medium">xA</th>
                <th className="label py-2 pl-3 font-medium">CS</th>
                <th className="label py-2 pl-3 font-medium">Range</th>
              </tr>
            </thead>
            <tbody>
              {upcoming.map((u) => (
                <tr key={u.gameweek} className="border-b border-border/60 text-right last:border-0">
                  <td className="num py-2 text-left text-muted">{u.gameweek}</td>
                  <td className="py-2 pl-2">
                    <FixtureCell club={club} clubs={byKey} gameweek={u.gameweek} lens={lens} />
                  </td>
                  <td className="py-2 pl-3">
                    <div className="flex items-center gap-2">
                      <span className="num w-8 text-left font-semibold">{u.xp.toFixed(1)}</span>
                      <span className="w-20">
                        <Bar value={u.xp} max={10} />
                      </span>
                    </div>
                  </td>
                  <td className="num py-2 pl-3 text-muted">{pct(u.p_60)}</td>
                  <td className="num py-2 pl-3 text-muted">{u.e_goals?.toFixed(2) ?? "–"}</td>
                  <td className="num py-2 pl-3 text-muted">{u.e_assists?.toFixed(2) ?? "–"}</td>
                  <td className="num py-2 pl-3 text-muted">{pct(u.p_clean_sheet)}</td>
                  <td className="num py-2 pl-3 text-muted">
                    {u.p10 != null ? `${u.p10}–${u.p90}` : "–"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {next && next.availability != null && next.availability < 1 ? (
            <p className="mt-3 text-xs text-muted">
              Expected points are scaled by FPL&apos;s availability flag ({pct(next.availability)}).
            </p>
          ) : null}
        </section>

        {/* Season */}
        <section className="card p-5">
          <SectionTitle aside={`${meta.season}`}>This season</SectionTitle>
          <div className="grid grid-cols-3 gap-x-4 gap-y-4">
            <Stat label="Points" value={s.points} hint={`${s.form.toFixed(1)} form`} />
            <Stat label="Minutes" value={s.minutes} hint={`${s.starts} starts`} />
            <Stat label="Bonus" value={s.bonus} />
            <Stat label="Goals" value={s.goals} hint={`xG ${s.xg.toFixed(1)}`} />
            <Stat label="Assists" value={s.assists} hint={`xA ${s.xa.toFixed(1)}`} />
            {p.position === "GKP" ? (
              <Stat label="Saves" value={s.saves} />
            ) : (
              <Stat label="Def. contrib." value={s.defensive_contribution} />
            )}
            <Stat label="xGI / 90" value={per90(s.xg + s.xa, s.minutes)} />
            <Stat label="Pts / 90" value={per90(s.points, s.minutes)} />
            <Stat label="Clean sheets" value={s.clean_sheets} />
          </div>
        </section>
      </div>

      <section className="card p-5">
        <SectionTitle aside="bars: points (green 10+) · dots: xG + xA">Points by gameweek</SectionTitle>
        <PointsChart matches={thisSeason} />
      </section>

      <section className="card p-5">
        <SectionTitle aside={`${thisSeason.length} matches`}>Match log · {meta.season}</SectionTitle>
        <MatchLog matches={thisSeason} clubs={byKey} />
      </section>

      <div className="grid gap-6 lg:grid-cols-2">
        {lastSeason.length ? (
          <section className="card p-5">
            <SectionTitle aside={lastSeason[0]?.season}>Last season</SectionTitle>
            <PointsChart matches={lastSeason} />
          </section>
        ) : null}
        {past_seasons.length ? (
          <section className="card p-5">
            <SectionTitle>Career in FPL</SectionTitle>
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border text-right">
                  {["Season", "Pts", "Min", "G", "A", "CS", "Max £"].map((h, i) => (
                    <th key={h} className={`label py-2 font-medium ${i ? "pl-3" : "text-left"}`}>
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {[...past_seasons].reverse().map((ps) => (
                  <tr key={ps.season} className="border-b border-border/60 text-right last:border-0">
                    <td className="num py-1.5 text-left text-muted">{ps.season}</td>
                    <td className="num py-1.5 pl-3 font-semibold">{ps.points}</td>
                    <td className="num py-1.5 pl-3">{ps.minutes}</td>
                    <td className="num py-1.5 pl-3">{ps.goals}</td>
                    <td className="num py-1.5 pl-3">{ps.assists}</td>
                    <td className="num py-1.5 pl-3">{ps.clean_sheets}</td>
                    <td className="num py-1.5 pl-3 text-muted">{ps.max_price.toFixed(1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
        ) : null}
      </div>
    </div>
  );
}

export default function Page(props: PageProps<"/players/[code]">) {
  return (
    <Suspense
      fallback={
        <div className="space-y-6">
          <Skeleton className="h-36" />
          <Skeleton className="h-72" />
        </div>
      }
    >
      <Profile params={props.params} />
    </Suspense>
  );
}
