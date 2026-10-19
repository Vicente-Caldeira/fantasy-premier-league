"use client";

import Image from "next/image";
import { useState } from "react";

import { photoUrl } from "@/lib/format";

/** Player headshot from the Premier League CDN; initials when there isn't one (new
 *  signings often have none for weeks). */
export function PlayerPhoto({ code, name, size = 40 }: { code: number; name: string; size?: number }) {
  const [failed, setFailed] = useState(false);
  const initials = name
    .split(/[\s.-]+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((w) => w[0]?.toUpperCase())
    .join("");
  return (
    // A fixed-size block whose children are all absolutely positioned: no in-flow text,
    // so the box and its baseline are identical with or without a photo.
    <span
      className="relative block shrink-0 overflow-hidden rounded-full bg-surface-2 align-middle text-muted"
      style={{ width: size, height: size, fontSize: Math.max(9, size / 3) }}
    >
      {failed ? (
        <span className="absolute inset-0 grid place-items-center leading-none font-semibold">{initials}</span>
      ) : (
        <Image
          src={photoUrl(code)}
          alt=""
          width={110}
          height={140}
          onError={() => setFailed(true)}
          className="absolute inset-x-0 top-0 h-auto w-full object-cover object-top"
          style={{ transform: "translateY(4%)" }}
        />
      )}
    </span>
  );
}
