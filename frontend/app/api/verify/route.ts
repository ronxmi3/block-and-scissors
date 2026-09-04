import { NextResponse } from 'next/server';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function POST(request: Request) {
  const backendUrl = process.env.PYTHON_BACKEND_URL ?? 'http://127.0.0.1:8000';
  const oracleKey = process.env.ORACLE_API_KEY;

  if (!oracleKey) {
    return NextResponse.json(
      { detail: 'Server is missing ORACLE_API_KEY in .env.local.' },
      { status: 500 },
    );
  }

  const incoming = await request.formData();
  const escrowId = incoming.get('escrow_id');
  const reference = incoming.get('reference');
  const result = incoming.get('result');
  const dryRun = incoming.get('dry_run') ?? 'true';

  if (typeof escrowId !== 'string' || !(reference instanceof File) || !(result instanceof File)) {
    return NextResponse.json(
      { detail: 'escrow_id, reference, and result are required.' },
      { status: 400 },
    );
  }

  const outgoing = new FormData();
  outgoing.set('escrow_id', escrowId);
  outgoing.set('reference', reference, reference.name);
  outgoing.set('result', result, result.name);
  outgoing.set('dry_run', String(dryRun));

  try {
    const response = await fetch(`${backendUrl}/evaluate-and-resolve`, {
      method: 'POST',
      headers: {
        'X-Oracle-Key': oracleKey,
      },
      body: outgoing,
      cache: 'no-store',
    });

    const text = await response.text();
    const contentType = response.headers.get('content-type') ?? 'application/json';

    // The Move contract uses abort code 2 for an escrow that has already been
    // resolved. Translate that expected contract rejection into a useful web
    // response instead of surfacing it as a generic 500 error.
    const abortCodeTwo =
      text.includes('resolve_escrow') &&
      text.includes('MoveAbort') &&
      text.includes('}, 2)');
    const alreadyResolved =
      !response.ok && (text.includes('E_ALREADY_RESOLVED') || abortCodeTwo);

    if (alreadyResolved) {
      return NextResponse.json(
        { detail: 'This escrow has already been resolved. Create a new escrow for another verification.' },
        { status: 409 },
      );
    }

    return new Response(text, {
      status: response.status,
      headers: { 'content-type': contentType },
    });
  } catch (error) {
    return NextResponse.json(
      {
        detail:
          error instanceof Error
            ? `Could not reach Python backend: ${error.message}`
            : 'Could not reach Python backend.',
      },
      { status: 502 },
    );
  }
}
