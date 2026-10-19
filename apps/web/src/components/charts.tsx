import type { MatchRow } from "@/lib/types";

/** Points per gameweek (bars) with expected goal involvement (xG + xA, dots). */
export function PointsChart({ matches, height = 140, slots = 38 }: { matches: MatchRow[]; height?: number; slots?: number }) {
  // One bar per gameweek (double gameweeks summed).
  const byGw = new Map<number, { points: number; xgi: number; minutes: number; label: string }>();
  for (const m of matches) {
    const prev = byGw.get(m.gameweek) ?? { points: 0, xgi: 0, minutes: 0, label: "" };
    byGw.set(m.gameweek, {
      points: prev.points + m.points,
      xgi: prev.xgi + (m.xg ?? 0) + (m.xa ?? 0),
      minutes: prev.minutes + m.minutes,
      label: `${prev.label ? `${prev.label}, ` : ""}${m.opponent.slice(0, 3).toUpperCase()} (${m.is_home ? "H" : "A"})`,
    });
  }
  if (byGw.size === 0) return <p className="text-sm text-muted">No matches yet this season.</p>;
  // A fixed season-long axis, so a few weeks don't stretch into giant bars.
  const weeks = Array.from({ length: slots }, (_, i) => i + 1).map(
    (gw) => [gw, byGw.get(gw) ?? { points: 0, xgi: 0, minutes: -1, label: "" }] as const,
  );
  const maxPoints = Math.max(10, ...weeks.map(([, w]) => w.points));
  const maxXgi = Math.max(1, ...weeks.map(([, w]) => w.xgi));
  const width = 100 / weeks.length;
  return (
    <div>
      <svg viewBox={`0 0 100 ${height}`} preserveAspectRatio="none" className="h-[140px] w-full overflow-visible">
        {[0.25, 0.5, 0.75].map((f) => (
          <line key={f} x1={0} x2={100} y1={height * f} y2={height * f} stroke="var(--border)" strokeWidth={0.3} vectorEffect="non-scaling-stroke" />
        ))}
        {weeks.map(([gw, w], i) => {
          const h = (Math.max(w.points, 0) / maxPoints) * (height - 4);
          return (
            <g key={gw}>
              <title>{w.minutes < 0 ? `GW${gw}` : `GW${gw} vs ${w.label}: ${w.points} pts, ${w.minutes}′, xGI ${w.xgi.toFixed(2)}`}</title>
              <rect
                x={i * width + width * 0.15}
                width={width * 0.7}
                y={height - h}
                height={Math.max(h, 0.8)}
                rx={0.6}
                fill={w.minutes <= 0 ? "var(--border)" : w.points >= 10 ? "var(--accent)" : "var(--faint)"}
                opacity={w.points >= 10 ? 1 : 0.55}
              />
            </g>
          );
        })}
      </svg>
      <div className="relative -mt-[140px] h-[140px]">
        {weeks.map(([gw, w], i) => w.minutes < 0 ? null : (
          <span
            key={gw}
            className="absolute h-1.5 w-1.5 -translate-x-1/2 translate-y-1/2 rounded-full bg-text ring-2 ring-surface"
            style={{ left: `${(i + 0.5) * width}%`, bottom: `${(w.xgi / maxXgi) * 92}%` }}
            title={`xGI ${w.xgi.toFixed(2)}`}
          />
        ))}
      </div>
      <div className="mt-1 flex text-[10px] text-faint">
        {weeks.map(([gw]) => (
          <span key={gw} className="num text-center" style={{ width: `${width}%` }}>
            {gw % 2 ? "" : gw}
          </span>
        ))}
      </div>
    </div>
  );
}
