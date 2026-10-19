import type { Club, ClubDetail, Fixture, Meta, Player, PlayerDetail } from "./types";

// Server components call the API directly; the browser goes through the /api rewrite.
const SERVER_API = process.env.API_URL ?? "http://127.0.0.1:8000";

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

async function get<T>(path: string): Promise<T> {
  const response = await fetch(`${SERVER_API}/api${path}`);
  if (!response.ok) {
    throw new ApiError(response.status, `${path}: ${response.status} ${await response.text()}`);
  }
  return (await response.json()) as T;
}

export const api = {
  meta: () => get<Meta>("/meta"),
  players: () => get<Player[]>("/players"),
  player: (code: number) => get<PlayerDetail>(`/players/${code}`),
  clubs: () => get<Club[]>("/clubs"),
  club: (key: string) => get<ClubDetail>(`/clubs/${key}`),
  fixtures: () => get<Fixture[]>("/fixtures"),
};
