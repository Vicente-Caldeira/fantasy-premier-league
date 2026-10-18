"use client";

import { PlayerPhoto } from "@/components/photo";
import { abbr, POSITIONS, venue } from "@/lib/format";
import type { Club, Player } from "@/lib/types";

export interface PitchSlot {
  player: Player;
  xp: number;
  captain: boolean;
  vice: boolean;
  isNew: boolean;
}

function fixtureLabel(player: Player, clubs: Record<string, Club>, gameweek: number) {
  const fixtures = clubs[player.team_key]?.fixtures.filter((f) => f.gameweek === gameweek) ?? [];
  if (fixtures.length === 0) return "Blank";
  return fixtures
    .map((f) => `${abbr(clubs, f.opponent)} (${venue(f.is_home)})`)
    .join(", ");
}

function Card({
  slot,
  clubs,
  gameweek,
  selected,
  onSelect,
  bench = false,
}: {
  slot: PitchSlot;
  clubs: Record<string, Club>;
  gameweek: number;
  selected: boolean;
  onSelect: () => void;
  bench?: boolean;
}) {
  const { player: p } = slot;
  const flagged = p.status !== "a";
  return (
    <button
      onClick={onSelect}
      className={`group relative flex w-[92px] flex-col items-center rounded-xl px-1 pt-1.5 pb-1 transition sm:w-[104px] ${
        selected ? "bg-white/25 ring-2 ring-white" : "hover:bg-white/10"
      } ${bench ? "opacity-90" : ""}`}
    >
      <span className="relative block">
        <PlayerPhoto code={p.code} name={p.name} size={46} />
        {slot.captain || slot.vice ? (
          <span
            className={`absolute -top-1 -right-2 grid h-5 w-5 place-items-center rounded-full text-[10px] font-bold ring-2 ring-pitch ${
              slot.captain ? "bg-black text-white" : "bg-white text-black"
            }`}
          >
            {slot.captain ? "C" : "V"}
          </span>
        ) : null}
        {flagged ? (
          <span
            title={p.news || "Flagged"}
            className={`absolute -top-1 -left-2 grid h-5 w-5 place-items-center rounded-full text-[10px] font-bold text-black ring-2 ring-pitch ${
              p.status === "d" ? "bg-warn" : "bg-danger"
            }`}
          >
            !
          </span>
        ) : null}
      </span>
      <span
        className={`mt-1 w-full truncate rounded-t-md px-1 py-0.5 text-center text-[11px] font-semibold ${
          slot.isNew ? "bg-accent text-black" : "bg-white text-black"
        }`}
      >
        {p.name}
      </span>
      <span className="w-full truncate rounded-b-md bg-black/55 px-1 py-0.5 text-center text-[10px] text-white">
        {fixtureLabel(p, clubs, gameweek)}
      </span>
      <span className="num mt-0.5 text-sm font-semibold text-white drop-shadow">
        {(slot.captain ? slot.xp * 2 : slot.xp).toFixed(1)}
      </span>
    </button>
  );
}

export function Pitch({
  lineup,
  bench,
  clubs,
  gameweek,
  selected,
  onSelect,
}: {
  lineup: PitchSlot[];
  bench: PitchSlot[];
  clubs: Record<string, Club>;
  gameweek: number;
  selected: number | null;
  onSelect: (code: number) => void;
}) {
  return (
    <div className="overflow-hidden rounded-xl">
      <div
        className="relative bg-pitch px-2 pt-4 pb-3"
        style={{
          backgroundImage:
            "repeating-linear-gradient(0deg, transparent 0 56px, rgba(255,255,255,0.035) 56px 112px)",
        }}
      >
        {/* Pitch markings */}
        <div className="pointer-events-none absolute inset-3 rounded-sm border border-pitch-line/70" />
        <div className="pointer-events-none absolute top-3 left-1/2 h-16 w-48 -translate-x-1/2 border border-t-0 border-pitch-line/70" />
        <div className="pointer-events-none absolute bottom-3 left-1/2 h-12 w-56 -translate-x-1/2 rounded-t-full border border-b-0 border-pitch-line/50" />
        <div className="relative flex flex-col gap-3 py-2">
          {POSITIONS.map((position) => (
            <div key={position} className="flex justify-center gap-1 sm:gap-3">
              {lineup
                .filter((s) => s.player.position === position)
                .map((s) => (
                  <Card
                    key={s.player.code}
                    slot={s}
                    clubs={clubs}
                    gameweek={gameweek}
                    selected={selected === s.player.code}
                    onSelect={() => onSelect(s.player.code)}
                  />
                ))}
            </div>
          ))}
        </div>
      </div>
      <div className="flex items-center justify-center gap-1 bg-pitch/80 px-2 py-2 sm:gap-3">
        <span className="label mr-1 hidden text-white/70 sm:inline">Bench</span>
        {bench.map((s) => (
          <Card
            key={s.player.code}
            slot={s}
            clubs={clubs}
            gameweek={gameweek}
            selected={selected === s.player.code}
            onSelect={() => onSelect(s.player.code)}
            bench
          />
        ))}
      </div>
    </div>
  );
}
