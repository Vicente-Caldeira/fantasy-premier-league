"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";

const LINKS = [
  { href: "/", label: "My team" },
  { href: "/players", label: "Players" },
  { href: "/fixtures", label: "Fixtures" },
  { href: "/clubs", label: "Clubs" },
  { href: "/compare", label: "Compare" },
  { href: "/chips", label: "Chips" },
];

function ActivePath({ children }: { children: (path: string) => React.ReactNode }) {
  return children(usePathname());
}

/** Nav links; `static` renders them without the active state (the prerendered shell). */
export function NavLinks({ static: isStatic = false }: { static?: boolean }) {
  return isStatic ? <Links path="" /> : <ActivePath>{(path) => <Links path={path} />}</ActivePath>;
}

function Links({ path }: { path: string }) {
  return (
    <nav className="flex items-center gap-1 overflow-x-auto text-sm">
      {LINKS.map(({ href, label }) => {
        const active = href === "/" ? path === "/" : path.startsWith(href);
        return (
          <Link
            key={href}
            href={href}
            className={`rounded-md px-2.5 py-1.5 whitespace-nowrap transition-colors ${
              active ? "bg-surface-2 font-medium text-text" : "text-muted hover:text-text"
            }`}
          >
            {label}
          </Link>
        );
      })}
    </nav>
  );
}

function remaining(deadline: Date, now: number) {
  const ms = deadline.getTime() - now;
  if (ms <= 0) return "passed";
  const d = Math.floor(ms / 86_400_000);
  const h = Math.floor((ms % 86_400_000) / 3_600_000);
  const m = Math.floor((ms % 3_600_000) / 60_000);
  return d > 0 ? `${d}d ${h}h` : h > 0 ? `${h}h ${m}m` : `${m}m`;
}

export function Deadline({ gameweek, deadline }: { gameweek: number | null; deadline: string | null }) {
  const [now, setNow] = useState<number | null>(null);
  useEffect(() => {
    // The first tick runs right away (not during render): the server can't know "now".
    const tick = () => setNow(Date.now());
    const first = setTimeout(tick, 0);
    const id = setInterval(tick, 30_000);
    return () => {
      clearTimeout(first);
      clearInterval(id);
    };
  }, []);
  if (!gameweek || !deadline) return null;
  const date = new Date(deadline);
  return (
    <div
      className="hidden items-center gap-2 rounded-md border border-border px-2.5 py-1 text-xs sm:flex"
      title={date.toLocaleString()}
    >
      <span className="text-muted">GW{gameweek} deadline</span>
      <span className="num font-medium">{now == null ? "…" : remaining(date, now)}</span>
    </div>
  );
}
