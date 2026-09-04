import { NextResponse } from 'next/server';

export const dynamic = 'force-dynamic';

export async function GET() {
  const backendUrl = process.env.PYTHON_BACKEND_URL ?? 'http://127.0.0.1:8000';

  try {
    const response = await fetch(`${backendUrl}/health`, {
      cache: 'no-store',
      signal: AbortSignal.timeout(5000),
    });

    const data = await response.json();
    return NextResponse.json(data, { status: response.status });
  } catch (error) {
    return NextResponse.json(
      {
        backend: 'offline',
        error: error instanceof Error ? error.message : 'Backend unavailable',
      },
      { status: 503 },
    );
  }
}
