// Browser-side API access (same-origin /api, rewritten to the backend).

export const ENTRY_KEY = "xpts.entry";

export async function post<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(`/api${path}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) throw new Error((await response.json().catch(() => null))?.detail ?? response.statusText);
  return (await response.json()) as T;
}

export async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(`/api${path}`);
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return (await response.json()) as T;
}

export function storedEntry(): number | null {
  try {
    const value = localStorage.getItem(ENTRY_KEY);
    return value ? Number(value) : null;
  } catch {
    return null;
  }
}
