import { Suspense } from "react";

import { MyTeam } from "@/components/my-team/my-team";
import { Skeleton } from "@/components/ui";
import { api } from "@/lib/api";

async function Team() {
  const [players, clubs, meta] = await Promise.all([api.players(), api.clubs(), api.meta()]);
  return <MyTeam players={players} clubs={clubs} meta={meta} />;
}

export default function Home() {
  return (
    <Suspense
      fallback={
        <div className="grid gap-6 lg:grid-cols-[1fr_380px]">
          <Skeleton className="h-[640px]" />
          <Skeleton className="h-[400px]" />
        </div>
      }
    >
      <Team />
    </Suspense>
  );
}
