import type { Metadata } from "next";
import { Suspense } from "react";

import { FixtureTicker } from "@/components/fixture-ticker";
import { Skeleton } from "@/components/ui";
import { api } from "@/lib/api";

export const metadata: Metadata = { title: "Fixtures" };

async function Ticker() {
  const [clubs, fixtures, meta] = await Promise.all([api.clubs(), api.fixtures(), api.meta()]);
  const upcoming = [...new Set(fixtures.filter((f) => f.home_goals == null).map((f) => f.gameweek))]
    .filter((gw) => gw >= (meta.next_gameweek ?? 0))
    .sort((a, b) => a - b);
  return <FixtureTicker clubs={clubs} fixtures={fixtures} gameweeks={upcoming} />;
}

export default function Page() {
  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-xl font-semibold tracking-tight">Fixtures</h1>
        <p className="mt-1 text-sm text-muted">
          Fixture difficulty from the model, not a 1–5 guess: expected goals, clean-sheet and win chances per match.
        </p>
      </div>
      <Suspense fallback={<Skeleton className="h-[720px]" />}>
        <Ticker />
      </Suspense>
    </div>
  );
}
