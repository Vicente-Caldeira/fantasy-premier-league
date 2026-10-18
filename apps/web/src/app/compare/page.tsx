import type { Metadata } from "next";
import { Suspense } from "react";

import { Compare } from "@/components/compare";
import { Skeleton } from "@/components/ui";
import { api } from "@/lib/api";

export const metadata: Metadata = { title: "Compare" };

async function Comparison() {
  const [players, clubs, meta] = await Promise.all([api.players(), api.clubs(), api.meta()]);
  return <Compare players={players} clubs={clubs} horizon={meta.horizon} />;
}

export default function Page() {
  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-xl font-semibold tracking-tight">Compare players</h1>
        <p className="mt-1 text-sm text-muted">Side by side over the next gameweeks and the season so far. The best value in each row is highlighted.</p>
      </div>
      <Suspense fallback={<Skeleton className="h-[640px]" />}>
        <Comparison />
      </Suspense>
    </div>
  );
}
