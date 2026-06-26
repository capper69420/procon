const API_BASE = import.meta.env.VITE_API_URL || "";

export class ApiError extends Error {
  status: number;
  details: unknown;

  constructor(message: string, status: number, details: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.details = details;
  }
}

async function parseBody(response: Response) {
  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    return response.json();
  }
  return response.text();
}

export async function apiRequest<T>(
  path: string,
  options: RequestInit = {},
  retries = 1
): Promise<T> {
  const url = `${API_BASE}${path}`;
  try {
    const response = await fetch(url, {
      ...options,
      headers: {
        ...(options.body instanceof FormData ? {} : { "Content-Type": "application/json" }),
        ...(options.headers || {}),
      },
    });
    const body = await parseBody(response).catch(() => null);
    if (!response.ok) {
      const message =
        typeof body === "object" && body && "detail" in body
          ? String((body as { detail: unknown }).detail)
          : `HTTP ${response.status}`;
      throw new ApiError(message, response.status, body);
    }
    return body as T;
  } catch (error) {
    if (retries > 0 && !(error instanceof ApiError && error.status < 500)) {
      await new Promise((resolve) => setTimeout(resolve, 350));
      return apiRequest<T>(path, options, retries - 1);
    }
    throw error;
  }
}

export async function checkHealth() {
  return apiRequest<{ status: string; supabase_connected: boolean }>("/health", {}, 0);
}
