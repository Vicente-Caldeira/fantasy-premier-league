import type { Metadata } from "next";
import { Suspense } from "react";

import { ChipPlanner } from "@/components/chip-planner";
import { Skeleton } from "@/components/ui";
import { api } from "@/lib/api";

export const metadata: Metadata = { title: "Chips" };

async function Planner() {
  const [players, clubs, meta] = await Promise.all([api.players(), api.clubs(), api.meta()]);
  return <ChipPlanner players={players} clubs={clubs} meta={meta} />;
}

export default function Page() {
  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-xl font-semibold tracking-tight">Chip planner</h1>
        <p className="mt-1 text-sm text-muted">
          What each chip would add to your team in the coming gameweeks, which ones you still have, and where the
          blanks and doubles fall.
        </p>
      </div>
      <Suspense fallback={<Skeleton className="h-[640px]" />}>
        <Planner />
      </Suspense>
    </div>
  );
}
