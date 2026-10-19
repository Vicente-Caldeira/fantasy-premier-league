import Image from "next/image";
import Link from "next/link";

export { PlayerPhoto } from "@/components/photo";

import {
  badgeUrl,
  difficulty,
  difficultyStyle,
  fixturesByGameweek,
  type Lens,
  pct,
  STATUS_LABEL,
} from "@/lib/format";
import type { Club, Player } from "@/lib/types";

export function ClubBadge({ club, size = 20 }: { club: Club | undefined; size?: number }) {
  if (!club) return <span style={{ width: size, height: size }} className="inline-block" />;
  return (
    <Image
      src={badgeUrl(club.fpl_team_code)}
      alt={club.name}
      width={size}
      height={size}
      className="inline-block shrink-0 object-contain"
    />
  );
}

/** Colour-coded opponents for one gameweek: "MCI" (home) / "mci" (away), blank = "–". */
export function FixtureCell({
  club,
  clubs,
  gameweek,
  lens,
  compact = false,
}: {
  club: Club | undefined;
  clubs: Record<string, Club>;
  gameweek: number;
  lens: Lens;
  compact?: boolean;
}) {
  const fixtures = fixturesByGameweek(club, [gameweek])[0];
  if (fixtures.length === 0) {
    return (
      <span className="flex h-6 items-center justify-center rounded text-[11px] text-faint ring-1 ring-border ring-inset">
        –
      </span>
    );
  }
  return (
    <span className="flex h-6 gap-px overflow-hidden rounded">
      {fixtures.map((f) => {
        const d = difficulty(f, lens);
        const short = clubs[f.opponent]?.short_name ?? f.opponent.slice(0, 3).toUpperCase();
        const title = `${clubs[f.opponent]?.name ?? f.opponent} (${f.is_home ? "home" : "away"}) · ${
          lens === "attack" ? `${f.goals_for.toFixed(2)} xG for` : `${pct(f.clean_sheet)} clean sheet`
        }`;
        return (
          <span
            key={`${f.gameweek}-${f.opponent}`}
            title={title}
            style={difficultyStyle(d)}
            className={`flex flex-1 items-center justify-center font-medium tracking-tight ${
              compact ? "px-0.5 text-[10px]" : "px-1 text-[11px]"
            }`}
          >
            {f.is_home ? short : short.toLowerCase()}
          </span>
        );
      })}
    </span>
  );
}

export function StatusPill({ player }: { player: Pick<Player, "status" | "news" | "chance_of_playing"> }) {
  if (player.status === "a") return null;
  const tone =
    player.status === "d" ? "bg-warn/20 text-warn" : "bg-danger/15 text-danger";
  return (
    <span
      title={player.news || STATUS_LABEL[player.status]}
      className={`inline-flex items-center rounded px-1.5 py-0.5 text-[10px] font-semibold ${tone}`}
    >
      {player.chance_of_playing != null ? `${player.chance_of_playing}%` : STATUS_LABEL[player.status]}
    </span>
  );
}

export function PriceMove({ player }: { player: Pick<Player, "p_rise" | "p_fall"> }) {
  const rise = player.p_rise ?? 0;
  const fall = player.p_fall ?? 0;
  if (rise < 0.15 && fall < 0.15) return <span className="text-faint">·</span>;
  return rise >= fall ? (
    <span className="text-accent-text" title={`${pct(rise)} chance of a rise before the next deadline`}>
      ▲{pct(rise)}
    </span>
  ) : (
    <span className="text-danger" title={`${pct(fall)} chance of a fall before the next deadline`}>
      ▼{pct(fall)}
    </span>
  );
}

export function Stat({
  label,
  value,
  hint,
  emphasis = false,
}: {
  label: string;
  value: React.ReactNode;
  hint?: React.ReactNode;
  emphasis?: boolean;
}) {
  return (
    <div className="min-w-0">
      <div className="label truncate">{label}</div>
      <div className={`num mt-1 truncate ${emphasis ? "text-2xl font-semibold" : "text-lg font-medium"}`}>
        {value}
      </div>
      {hint ? <div className="mt-0.5 truncate text-xs text-muted">{hint}</div> : null}
    </div>
  );
}

export function Bar({ value, max, tone = "accent" }: { value: number; max: number; tone?: "accent" | "muted" }) {
  const width = `${Math.max(0, Math.min(1, value / max)) * 100}%`;
  return (
    <span className="block h-1.5 w-full overflow-hidden rounded-full bg-surface-2">
      <span
        className={`block h-full rounded-full ${tone === "accent" ? "bg-accent" : "bg-faint"}`}
        style={{ width }}
      />
    </span>
  );
}

/** Win / draw / loss as one stacked bar from the given side's view. */
export function OutcomeBar({ win, draw, loss }: { win: number; draw: number; loss: number }) {
  return (
    <span className="flex h-1.5 w-full overflow-hidden rounded-full" title={`W ${pct(win)} · D ${pct(draw)} · L ${pct(loss)}`}>
      <span className="bg-accent" style={{ width: pct(win, 1) }} />
      <span className="bg-faint/60" style={{ width: pct(draw, 1) }} />
      <span className="bg-danger/80" style={{ width: pct(loss, 1) }} />
    </span>
  );
}

export function SectionTitle({ children, aside }: { children: React.ReactNode; aside?: React.ReactNode }) {
  return (
    <div className="mb-3 flex items-baseline justify-between gap-4">
      <h2 className="text-sm font-semibold tracking-tight">{children}</h2>
      {aside ? <div className="text-xs text-muted">{aside}</div> : null}
    </div>
  );
}

export function PlayerLink({ code, children, className = "" }: { code: number; children: React.ReactNode; className?: string }) {
  return (
    <Link href={`/players/${code}`} className={`hover:text-accent-text ${className}`}>
      {children}
    </Link>
  );
}

export function Skeleton({ className = "" }: { className?: string }) {
  return <div className={`animate-pulse rounded-lg bg-surface-2 ${className}`} />;
}
