import type { Metadata } from "next";
import { Suspense } from "react";

import { PlayersTable } from "@/components/players-table";
import { Skeleton } from "@/components/ui";
import { api } from "@/lib/api";

export const metadata: Metadata = { title: "Players" };

async function Players() {
  const [players, clubs, meta] = await Promise.all([api.players(), api.clubs(), api.meta()]);
  return <PlayersTable players={players} clubs={clubs} horizon={meta.horizon} />;
}

export default function Page() {
  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-xl font-semibold tracking-tight">Players</h1>
        <p className="mt-1 text-sm text-muted">
          Expected points per gameweek with uncertainty, form, ownership and tonight&apos;s price-move odds.
        </p>
      </div>
      <Suspense fallback={<Skeleton className="h-[640px]" />}>
        <Players />
      </Suspense>
    </div>
  );
}
