export const PACKAGE_ID =
  process.env.NEXT_PUBLIC_SUI_PACKAGE_ID ??
  '0x5d7fa930a5d95ae7f8a2a56693e5341cb8e84dd0865fa42a6aac215c2659057a';

export const DEFAULT_BARBER =
  process.env.NEXT_PUBLIC_DEFAULT_BARBER ??
  '0x0a9f562565702fc0b81a83a1bd42fdc1ec13c4548b3f1e90e4e62f5aeae15fc9';

export const DEFAULT_THRESHOLD = Number(
  process.env.NEXT_PUBLIC_DEFAULT_THRESHOLD ?? '80',
);

export const ESCROW_TYPE = `${PACKAGE_ID}::haircut_escrow::HaircutEscrow`;

export function isObjectId(value: string) {
  return /^0x[0-9a-fA-F]{64}$/.test(value.trim());
}

export function suiToMist(value: string): bigint {
  const input = value.trim();
  if (!/^\d+(\.\d{0,9})?$/.test(input)) {
    throw new Error('Enter a valid SUI amount with at most 9 decimal places.');
  }

  const [whole, fraction = ''] = input.split('.');
  const padded = `${fraction}000000000`.slice(0, 9);
  return BigInt(whole) * 1_000_000_000n + BigInt(padded || '0');
}

export function shortId(value?: string | null, start = 8, end = 6) {
  if (!value) return '—';
  if (value.length <= start + end + 3) return value;
  return `${value.slice(0, start)}…${value.slice(-end)}`;
}

/**
 * dApp Kit's execution result intentionally stays small. We read the transaction
 * back with objectTypes enabled, then locate the newly-created HaircutEscrow.
 * This recursive fallback is tolerant of small SDK response-shape changes.
 */
export function findEscrowObjectId(value: unknown): string | null {
  const seen = new Set<object>();

  function walk(node: unknown): string | null {
    if (!node || typeof node !== 'object') return null;
    if (seen.has(node as object)) return null;
    seen.add(node as object);

    if (Array.isArray(node)) {
      for (const item of node) {
        const result = walk(item);
        if (result) return result;
      }
      return null;
    }

    const record = node as Record<string, unknown>;

    // Common shape: objectTypes: { "0xOBJECT": "0xPKG::module::Type" }
    for (const [key, child] of Object.entries(record)) {
      if (isObjectId(key) && typeof child === 'string' && child === ESCROW_TYPE) {
        return key;
      }
    }

    // Other common shapes place objectId and type in the same object.
    if (typeof record.objectId === 'string' && isObjectId(record.objectId)) {
      const type =
        typeof record.type === 'string'
          ? record.type
          : typeof record.objectType === 'string'
            ? record.objectType
            : null;
      if (type === ESCROW_TYPE) return record.objectId;
    }

    for (const child of Object.values(record)) {
      const result = walk(child);
      if (result) return result;
    }
    return null;
  }

  return walk(value);
}
