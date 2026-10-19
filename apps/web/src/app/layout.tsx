import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import Link from "next/link";
import { Suspense } from "react";

import { Deadline, NavLinks } from "@/components/nav";
import { api } from "@/lib/api";

import "./globals.css";

const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });

export const metadata: Metadata = {
  title: { default: "xPts", template: "%s · xPts" },
  description: "Expected points, fixtures and transfer planning for Fantasy Premier League",
};

async function HeaderMeta() {
  const meta = await api.meta().catch(() => null);
  if (!meta) return <span className="text-xs text-danger">API offline</span>;
  return <Deadline gameweek={meta.next_gameweek} deadline={meta.next_deadline} />;
}

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}>
      <body className="flex min-h-full flex-col">
        <header className="sticky top-0 z-30 border-b border-border bg-bg/85 backdrop-blur">
          <div className="mx-auto flex h-14 w-full max-w-7xl items-center gap-4 px-4">
            <Link href="/" className="flex items-center gap-2 font-semibold tracking-tight">
              <span className="grid h-6 w-6 place-items-center rounded-md bg-accent text-[11px] font-bold text-white dark:text-black">
                xP
              </span>
              <span className="hidden sm:inline">xPts</span>
            </Link>
            <Suspense fallback={<NavLinks static />}>
              <NavLinks />
            </Suspense>
            <div className="ml-auto">
              <Suspense fallback={null}>
                <HeaderMeta />
              </Suspense>
            </div>
          </div>
        </header>
        <main className="mx-auto w-full max-w-7xl flex-1 px-4 py-6">{children}</main>
        <footer className="border-t border-border py-4 text-center text-xs text-faint">
          Predictions from a walk-forward-validated component model · not affiliated with the Premier League
        </footer>
      </body>
    </html>
  );
}
