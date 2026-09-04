'use client';

import {
  useCurrentAccount,
  useCurrentClient,
  useCurrentNetwork,
  useDAppKit,
} from '@mysten/dapp-kit-react';
import { ConnectButton } from '@mysten/dapp-kit-react/ui';
import { Transaction } from '@mysten/sui/transactions';
import {
  ArrowRight,
  Check,
  CheckCircle2,
  Clipboard,
  ExternalLink,
  ImagePlus,
  Loader2,
  LockKeyhole,
  RefreshCcw,
  Scissors,
  ShieldCheck,
  Sparkles,
  WalletCards,
  X,
} from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';
import type { ChangeEvent, CSSProperties } from 'react';
import {
  DEFAULT_BARBER,
  DEFAULT_THRESHOLD,
  ESCROW_TYPE,
  PACKAGE_ID,
  findEscrowObjectId,
  isObjectId,
  shortId,
  suiToMist,
} from '@/lib/sui';

type VerificationResponse = {
  score?: number;
  raw_similarity?: number;
  threshold?: number;
  predicted_outcome?: 'BARBER_PAID' | 'CUSTOMER_REFUNDED' | string;
  regions?: Record<string, number>;
  sui?: {
    success?: boolean;
    dry_run?: boolean;
    transaction_digest?: string | null;
    status?: string | null;
    error?: string | null;
    stderr?: string | null;
  };
  detail?: string;
};

type BackendHealth = {
  backend?: string;
  sui_ok?: boolean;
  active_env?: string;
  threshold?: number;
};

type SettlementReceipt = {
  outcome: 'BARBER_PAID' | 'CUSTOMER_REFUNDED';
  score: number;
  threshold: number;
  amountSui: string;
  escrowId: string;
  customer: string;
  barber: string;
  recipient: string;
  transactionDigest: string;
  network: string;
  recordedAt: string;
  chainConfirmed: boolean;
};


type UploadCardProps = {
  label: string;
  hint: string;
  file: File | null;
  preview: string | null;
  onChange: (event: ChangeEvent<HTMLInputElement>) => void;
  onClear: () => void;
};

function UploadCard({ label, hint, file, preview, onChange, onClear }: UploadCardProps) {
  return (
    <div className="upload-card">
      <div className="upload-heading">
        <div>
          <span className="eyebrow">{label}</span>
          <p>{hint}</p>
        </div>
        {file ? (
          <button className="icon-button" type="button" onClick={onClear} aria-label="Remove image">
            <X size={16} />
          </button>
        ) : null}
      </div>

      <label className={`dropzone ${preview ? 'has-preview' : ''}`}>
        {preview ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={preview} alt={`${label} preview`} />
        ) : (
          <div className="dropzone-empty">
            <span className="dropzone-icon">
              <ImagePlus size={23} />
            </span>
            <strong>Choose image</strong>
            <small>JPG, PNG or WEBP</small>
          </div>
        )}
        <input type="file" accept="image/jpeg,image/png,image/webp" onChange={onChange} />
      </label>

      <div className="file-row">
        <span>{file?.name ?? 'No file selected'}</span>
        {file ? <Check size={15} /> : null}
      </div>
    </div>
  );
}

function copy(value: string) {
  void navigator.clipboard?.writeText(value);
}

export function HaircutDashboard() {
  const account = useCurrentAccount();
  const network = useCurrentNetwork();
  const client = useCurrentClient();
  const dAppKit = useDAppKit();

  const [barber, setBarber] = useState(DEFAULT_BARBER);
  const [amount, setAmount] = useState('0.1');
  const [threshold, setThreshold] = useState(DEFAULT_THRESHOLD);
  const [escrowId, setEscrowId] = useState('');
  const [createDigest, setCreateDigest] = useState('');
  const [creating, setCreating] = useState(false);
  const [createError, setCreateError] = useState('');

  const [reference, setReference] = useState<File | null>(null);
  const [result, setResult] = useState<File | null>(null);
  const [referencePreview, setReferencePreview] = useState<string | null>(null);
  const [resultPreview, setResultPreview] = useState<string | null>(null);
  const [dryRun, setDryRun] = useState(true);
  const [verifying, setVerifying] = useState(false);
  const [verifyError, setVerifyError] = useState('');
  const [verification, setVerification] = useState<VerificationResponse | null>(null);
  const [receipt, setReceipt] = useState<SettlementReceipt | null>(null);
  const [lockedAmount, setLockedAmount] = useState('');
  const [lockedBarber, setLockedBarber] = useState('');
  const [lockedCustomer, setLockedCustomer] = useState('');
  const verifyLock = useRef(false);

  const [health, setHealth] = useState<BackendHealth | null>(null);
  const [healthLoading, setHealthLoading] = useState(true);

  useEffect(() => {
    const savedEscrow = localStorage.getItem('blocks-and-scissors:escrow');
    const savedDigest = localStorage.getItem('blocks-and-scissors:create-digest');
    const savedAmount = localStorage.getItem('blocks-and-scissors:locked-amount');
    const savedBarber = localStorage.getItem('blocks-and-scissors:locked-barber');
    const savedCustomer = localStorage.getItem('blocks-and-scissors:locked-customer');
    const savedReceipt = localStorage.getItem('blocks-and-scissors:last-receipt');

    if (savedEscrow) setEscrowId(savedEscrow);
    if (savedDigest) setCreateDigest(savedDigest);
    if (savedAmount) setLockedAmount(savedAmount);
    if (savedBarber) setLockedBarber(savedBarber);
    if (savedCustomer) setLockedCustomer(savedCustomer);
    if (savedReceipt) {
      try {
        setReceipt(JSON.parse(savedReceipt) as SettlementReceipt);
      } catch {
        localStorage.removeItem('blocks-and-scissors:last-receipt');
      }
    }
  }, []);

  useEffect(() => {
    let cancelled = false;

    async function checkHealth() {
      setHealthLoading(true);
      try {
        const response = await fetch('/api/backend-health', { cache: 'no-store' });
        const body = (await response.json()) as BackendHealth;
        if (!cancelled) setHealth(body);
      } catch {
        if (!cancelled) setHealth({ backend: 'offline', sui_ok: false });
      } finally {
        if (!cancelled) setHealthLoading(false);
      }
    }

    void checkHealth();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    return () => {
      if (referencePreview) URL.revokeObjectURL(referencePreview);
      if (resultPreview) URL.revokeObjectURL(resultPreview);
    };
  }, [referencePreview, resultPreview]);

  const backendReady = health?.backend === 'ok' && health?.sui_ok === true;
  const score = verification?.score;
  const passed = verification?.predicted_outcome === 'BARBER_PAID';
  const settled = verification?.sui?.success === true && verification?.sui?.dry_run === false;
  const simulated = verification?.sui?.success === true && verification?.sui?.dry_run === true;
  const receiptMatchesEscrow = Boolean(receipt && escrowId && receipt.escrowId === escrowId.trim());
  const escrowSettled = settled || receiptMatchesEscrow;

  const progress = useMemo(() => {
    if (settled) return 3;
    if (verification) return 2;
    if (escrowId) return 1;
    return 0;
  }, [escrowId, verification, settled]);

  function changeImage(kind: 'reference' | 'result', file: File | null) {
    if (kind === 'reference') {
      if (referencePreview) URL.revokeObjectURL(referencePreview);
      setReference(file);
      setReferencePreview(file ? URL.createObjectURL(file) : null);
    } else {
      if (resultPreview) URL.revokeObjectURL(resultPreview);
      setResult(file);
      setResultPreview(file ? URL.createObjectURL(file) : null);
    }
    setVerification(null);
    setVerifyError('');
  }

  async function createEscrow() {
    setCreateError('');
    setVerification(null);
    setReceipt(null);
    localStorage.removeItem('blocks-and-scissors:last-receipt');

    if (!account) {
      setCreateError('Connect a Sui wallet first.');
      return;
    }
    if (network !== 'testnet') {
      setCreateError('This build is locked to Sui Testnet.');
      return;
    }
    if (!isObjectId(barber)) {
      setCreateError('Enter a valid 0x Sui barber address.');
      return;
    }
    if (threshold < 1 || threshold > 100) {
      setCreateError('Threshold must be between 1 and 100.');
      return;
    }

    try {
      setCreating(true);
      const paymentMist = suiToMist(amount);
      if (paymentMist <= 0n) throw new Error('Payment must be greater than 0 SUI.');

      const tx = new Transaction();
      const payment = tx.coin({ balance: paymentMist });

      tx.moveCall({
        target: `${PACKAGE_ID}::haircut_escrow::create_escrow`,
        arguments: [
          tx.pure.address(barber.trim()),
          payment,
          tx.pure.u8(threshold),
        ],
      });

      const execution: any = await dAppKit.signAndExecuteTransaction({ transaction: tx });
      if (execution.FailedTransaction) {
        throw new Error(
          execution.FailedTransaction.status?.error?.message ?? 'Sui transaction failed.',
        );
      }

      const digest = execution.Transaction?.digest;
      if (!digest) throw new Error('Wallet returned no transaction digest.');

      setCreateDigest(digest);
      localStorage.setItem('blocks-and-scissors:create-digest', digest);

      await client.waitForTransaction({ digest, timeout: 60_000 });
      const details: any = await client.getTransaction({
        digest,
        include: { effects: true, objectTypes: true },
      });

      const newEscrowId = findEscrowObjectId(details);
      if (!newEscrowId) {
        throw new Error(
          `The escrow transaction succeeded, but the app could not locate the ${ESCROW_TYPE} object. Transaction: ${digest}`,
        );
      }

      setEscrowId(newEscrowId);
      setLockedAmount(amount);
      setLockedBarber(barber.trim());
      setLockedCustomer(account.address);
      localStorage.setItem('blocks-and-scissors:escrow', newEscrowId);
      localStorage.setItem('blocks-and-scissors:locked-amount', amount);
      localStorage.setItem('blocks-and-scissors:locked-barber', barber.trim());
      localStorage.setItem('blocks-and-scissors:locked-customer', account.address);
    } catch (error) {
      setCreateError(error instanceof Error ? error.message : 'Could not create escrow.');
    } finally {
      setCreating(false);
    }
  }

  async function verifyHaircut() {
    setVerifyError('');

    if (verifyLock.current) return;
    if (escrowSettled) {
      setVerifyError('This escrow has already been settled. Start another haircut to create a new escrow.');
      return;
    }
    if (!isObjectId(escrowId)) {
      setVerifyError('Enter or create a valid HaircutEscrow object ID.');
      return;
    }
    if (!reference || !result) {
      setVerifyError('Choose both a reference and finished-haircut image.');
      return;
    }

    try {
      verifyLock.current = true;
      setVerifying(true);
      setVerification(null);

      const form = new FormData();
      form.set('escrow_id', escrowId.trim());
      form.set('reference', reference, reference.name);
      form.set('result', result, result.name);
      form.set('dry_run', String(dryRun));

      const response = await fetch('/api/verify', {
        method: 'POST',
        body: form,
      });

      const body = (await response.json()) as VerificationResponse;
      setVerification(body);

      if (!response.ok) {
        throw new Error(body.detail ?? `Verification failed (${response.status}).`);
      }
      if (body.sui?.success === false) {
        const detail = body.sui.error || body.sui.stderr || body.sui.status;
        throw new Error(detail || 'AI scored the images, but the Sui transaction failed.');
      }

      const liveSettlement = body.sui?.success === true && body.sui?.dry_run === false;
      const digest = body.sui?.transaction_digest;
      const outcome = body.predicted_outcome;

      if (
        liveSettlement &&
        digest &&
        (outcome === 'BARBER_PAID' || outcome === 'CUSTOMER_REFUNDED')
      ) {
        let chainConfirmed = false;
        try {
          await client.waitForTransaction({ digest, timeout: 60_000 });
          chainConfirmed = true;
        } catch {
          // The backend already returned a successful live transaction. If the wallet
          // client is briefly behind, keep the receipt and mark it as submitted.
        }

        const customer = lockedCustomer || account?.address || '';
        const barberAddress = lockedBarber || barber.trim();
        const newReceipt: SettlementReceipt = {
          outcome,
          score: body.score ?? 0,
          threshold: body.threshold ?? threshold,
          amountSui: lockedAmount || amount,
          escrowId: escrowId.trim(),
          customer,
          barber: barberAddress,
          recipient: outcome === 'BARBER_PAID' ? barberAddress : customer,
          transactionDigest: digest,
          network: 'Sui Testnet',
          recordedAt: new Date().toISOString(),
          chainConfirmed,
        };

        setReceipt(newReceipt);
        localStorage.setItem('blocks-and-scissors:last-receipt', JSON.stringify(newReceipt));
      }
    } catch (error) {
      setVerifyError(error instanceof Error ? error.message : 'Verification failed.');
    } finally {
      verifyLock.current = false;
      setVerifying(false);
    }
  }

  function clearEscrow() {
    setEscrowId('');
    setCreateDigest('');
    setVerification(null);
    setReceipt(null);
    setLockedAmount('');
    setLockedBarber('');
    setLockedCustomer('');
    setCreateError('');
    setVerifyError('');
    localStorage.removeItem('blocks-and-scissors:escrow');
    localStorage.removeItem('blocks-and-scissors:create-digest');
    localStorage.removeItem('blocks-and-scissors:locked-amount');
    localStorage.removeItem('blocks-and-scissors:locked-barber');
    localStorage.removeItem('blocks-and-scissors:locked-customer');
    localStorage.removeItem('blocks-and-scissors:last-receipt');
  }

  return (
    <main className="site-shell">
      <div className="ambient ambient-one" />
      <div className="ambient ambient-two" />

      <header className="topbar">
        <a className="brand" href="#top" aria-label="Blocks and Scissors home">
          <span className="brand-mark"><Scissors size={19} /></span>
          <span>
            <strong>Blocks</strong>
            <i>&</i>
            <strong>Scissors</strong>
          </span>
        </a>

        <div className="topbar-right">
          <span className={`network-pill ${backendReady ? 'online' : ''}`}>
            <span className="status-dot" />
            {healthLoading ? 'Checking oracle' : backendReady ? 'Oracle online' : 'Oracle offline'}
          </span>
          <ConnectButton><span>Connect wallet</span></ConnectButton>
        </div>
      </header>

      <section id="top" className="hero">
        <div>
          <span className="hero-kicker"><Sparkles size={14} /> AI-verified escrow on Sui</span>
          <h1>Trust the cut.<br /><em>Not the promise.</em></h1>
          <p className="hero-copy">
            Lock payment before the haircut. Let computer vision judge the result.
            Release SUI only when the cut clears the agreed similarity threshold.
          </p>
        </div>

        <aside className="hero-proof">
          <div className="proof-number">80<span>/100</span></div>
          <p>Default release threshold</p>
          <div className="proof-line" />
          <small>Testnet · non-custodial contract · oracle settlement</small>
        </aside>
      </section>

      <section className="stepper" aria-label="Escrow progress">
        {[
          ['01', 'Lock payment'],
          ['02', 'Verify haircut'],
          ['03', 'Settle on Sui'],
        ].map(([number, label], index) => (
          <div className={`step ${progress >= index + 1 ? 'done' : ''} ${progress === index ? 'active' : ''}`} key={number}>
            <span>{progress >= index + 1 ? <Check size={15} /> : number}</span>
            <strong>{label}</strong>
          </div>
        ))}
      </section>

      <section className="workspace">
        <article className="panel create-panel">
          <div className="panel-header">
            <div>
              <span className="eyebrow">Step 01</span>
              <h2>Create escrow</h2>
            </div>
            <span className="panel-icon"><LockKeyhole size={20} /></span>
          </div>

          <div className="wallet-state">
            <div>
              <span>Customer wallet</span>
              <strong>{account ? shortId(account.address, 10, 8) : 'Not connected'}</strong>
            </div>
            <div className={`tiny-badge ${account ? 'good' : ''}`}>
              {account ? 'Connected' : 'Required'}
            </div>
          </div>

          <label className="field">
            <span>Barber address</span>
            <input value={barber} onChange={(e) => setBarber(e.target.value)} spellCheck={false} />
          </label>

          <div className="field-grid">
            <label className="field">
              <span>Payment</span>
              <div className="input-unit">
                <input value={amount} onChange={(e) => setAmount(e.target.value)} inputMode="decimal" />
                <b>SUI</b>
              </div>
            </label>

            <label className="field">
              <span>Threshold</span>
              <div className="threshold-box">
                <strong>{threshold}</strong><small>/100</small>
              </div>
            </label>
          </div>

          <input
            className="range"
            type="range"
            min="1"
            max="100"
            value={threshold}
            onChange={(e) => setThreshold(Number(e.target.value))}
          />

          {createError ? <div className="error-box">{createError}</div> : null}

          <button className="primary-button" type="button" onClick={createEscrow} disabled={creating || !account}>
            {creating ? <Loader2 className="spin" size={18} /> : <WalletCards size={18} />}
            {creating ? 'Creating on Sui…' : 'Lock payment in escrow'}
            {!creating ? <ArrowRight size={18} /> : null}
          </button>

          <p className="microcopy">Your wallet signs the transaction. The payment is held by the shared Move object, not by this website.</p>
        </article>

        <article className="panel escrow-panel">
          <div className="panel-header">
            <div>
              <span className="eyebrow">Live object</span>
              <h2>Current escrow</h2>
            </div>
            <span className={`panel-icon ${escrowId ? 'green' : ''}`}><ShieldCheck size={20} /></span>
          </div>

          <div className={`escrow-state ${escrowId ? 'ready' : ''}`}>
            <div className="state-orb">{escrowId ? <CheckCircle2 size={27} /> : <LockKeyhole size={25} />}</div>
            <div>
              <strong>{escrowId ? (escrowSettled ? 'Escrow settled' : 'Escrow locked') : 'Waiting for escrow'}</strong>
              <p>{escrowId ? (escrowSettled ? 'Settlement complete. Receipt is available below.' : 'Ready for AI verification.') : 'Create one or paste an existing Testnet object ID.'}</p>
            </div>
          </div>

          <label className="field">
            <span>HaircutEscrow object ID</span>
            <div className="copy-input">
              <input
                value={escrowId}
                onChange={(e) => {
                  const nextEscrow = e.target.value;
                  setEscrowId(nextEscrow);
                  setVerification(null);
                  if (receipt && receipt.escrowId !== nextEscrow.trim()) {
                    setReceipt(null);
                    localStorage.removeItem('blocks-and-scissors:last-receipt');
                  }
                }}
                placeholder="0x…"
                spellCheck={false}
              />
              <button type="button" onClick={() => escrowId && copy(escrowId)} aria-label="Copy escrow ID"><Clipboard size={16} /></button>
            </div>
          </label>

          <div className="object-meta">
            <div><span>Network</span><strong>{network ?? 'testnet'}</strong></div>
            <div><span>Contract</span><strong>{shortId(PACKAGE_ID)}</strong></div>
            <div><span>Create tx</span><strong>{shortId(createDigest)}</strong></div>
          </div>

          {escrowId ? (
            <button className="ghost-button" type="button" onClick={clearEscrow}>
              <RefreshCcw size={16} /> Start another haircut
            </button>
          ) : null}
        </article>
      </section>

      <section className="verify-section">
        <div className="section-heading">
          <div>
            <span className="eyebrow">Step 02</span>
            <h2>AI haircut verification</h2>
          </div>
          <p>Reference vs. finished cut. Your Python oracle scores the pair, then submits the score to the Move contract.</p>
        </div>

        <div className="verify-grid">
          <UploadCard
            label="Reference"
            hint="The haircut that was agreed on."
            file={reference}
            preview={referencePreview}
            onChange={(e) => changeImage('reference', e.target.files?.[0] ?? null)}
            onClear={() => changeImage('reference', null)}
          />

          <div className="compare-mark"><Scissors size={20} /></div>

          <UploadCard
            label="Finished cut"
            hint="The haircut after the appointment."
            file={result}
            preview={resultPreview}
            onChange={(e) => changeImage('result', e.target.files?.[0] ?? null)}
            onClear={() => changeImage('result', null)}
          />
        </div>

        <div className="verify-action-row">
          <label className="switch-row">
            <input type="checkbox" checked={dryRun} onChange={(e) => setDryRun(e.target.checked)} />
            <span className="switch"><i /></span>
            <span>
              <strong>Simulation mode</strong>
              <small>{dryRun ? 'No SUI moves.' : 'LIVE settlement on Testnet.'}</small>
            </span>
          </label>

          <button
            className="verify-button"
            type="button"
            onClick={verifyHaircut}
            disabled={verifying || !reference || !result || !escrowId || escrowSettled}
          >
            {verifying ? <Loader2 className="spin" size={18} /> : escrowSettled ? <CheckCircle2 size={18} /> : <Sparkles size={18} />}
            {verifying
              ? 'Comparing haircuts…'
              : escrowSettled
                ? receipt?.outcome === 'CUSTOMER_REFUNDED'
                  ? 'Refunded ✓'
                  : 'Barber paid ✓'
                : dryRun
                  ? 'Run AI simulation'
                  : 'Verify & settle'}
          </button>
        </div>

        {verifyError ? <div className="error-box wide">{verifyError}</div> : null}
      </section>

      <section className={`result-panel ${verification ? 'visible' : ''} ${passed ? 'pass' : 'refund'}`}>
        {!verification ? (
          <div className="result-empty">
            <Sparkles size={22} />
            <p>Your AI verdict will appear here.</p>
          </div>
        ) : (
          <>
            <div className="score-wrap">
              <div className="score-ring" style={{ '--score': `${Math.max(0, Math.min(100, score ?? 0)) * 3.6}deg` } as CSSProperties}>
                <div><strong>{score ?? '—'}</strong><span>/100</span></div>
              </div>
              <div>
                <span className="eyebrow">AI verdict</span>
                <h2>{passed ? 'Haircut verified.' : 'Threshold not met.'}</h2>
                <p>
                  {passed
                    ? `Score ${score} clears the ${verification.threshold ?? threshold}-point release threshold.`
                    : `Score ${score} is below the ${verification.threshold ?? threshold}-point release threshold.`}
                </p>
              </div>
            </div>

            <div className="verdict-card">
              <span>Contract outcome</span>
              <strong>{passed ? 'BARBER PAID' : 'CUSTOMER REFUNDED'}</strong>
              <small>
                {simulated
                  ? 'Simulation passed — nothing moved.'
                  : settled
                    ? 'Settlement executed on Sui Testnet.'
                    : verification.sui?.success === false
                      ? 'Sui settlement failed.'
                      : 'Awaiting settlement status.'}
              </small>
            </div>

            <div className="result-stats">
              <div><span>Raw similarity</span><strong>{verification.raw_similarity?.toFixed?.(4) ?? '—'}</strong></div>
              <div><span>Threshold</span><strong>{verification.threshold ?? threshold}</strong></div>
              <div><span>Sui mode</span><strong>{verification.sui?.dry_run ? 'Dry run' : 'Live'}</strong></div>
              <div><span>Tx digest</span><strong>{shortId(verification.sui?.transaction_digest)}</strong></div>
            </div>

            {verification.sui?.transaction_digest ? (
              <a
                className="explorer-button"
                href={`https://suiscan.xyz/testnet/tx/${verification.sui.transaction_digest}`}
                target="_blank"
                rel="noreferrer"
              >
                View transaction <ExternalLink size={15} />
              </a>
            ) : null}
          </>
        )}
      </section>

      {receipt ? (
        <section className={`receipt-panel ${receipt.outcome === 'BARBER_PAID' ? 'paid' : 'refunded'}`}>
          <div className="receipt-top">
            <div>
              <span className="eyebrow">On-chain settlement receipt</span>
              <h2>{receipt.outcome === 'BARBER_PAID' ? 'Barber paid' : 'Customer refunded'}</h2>
              <p>Permanent proof of the settlement transaction on Sui Testnet.</p>
            </div>
            <div className={`receipt-status ${receipt.chainConfirmed ? 'confirmed' : ''}`}>
              <CheckCircle2 size={18} />
              <span>{receipt.chainConfirmed ? 'On-chain confirmed' : 'Transaction submitted'}</span>
            </div>
          </div>

          <div className="receipt-amount">
            <span>{receipt.outcome === 'BARBER_PAID' ? 'Released to barber' : 'Returned to customer'}</span>
            <strong>{receipt.amountSui || '—'} <small>SUI</small></strong>
          </div>

          <div className="receipt-grid">
            <div><span>AI score</span><strong>{receipt.score}/100</strong></div>
            <div><span>Threshold</span><strong>{receipt.threshold}/100</strong></div>
            <div><span>Network</span><strong>{receipt.network}</strong></div>
            <div><span>Outcome</span><strong>{receipt.outcome.replace('_', ' ')}</strong></div>
          </div>

          <div className="receipt-ledger">
            <div className="receipt-row">
              <span>Escrow object</span>
              <strong title={receipt.escrowId}>{shortId(receipt.escrowId, 12, 10)}</strong>
              <button type="button" onClick={() => copy(receipt.escrowId)} aria-label="Copy escrow object ID"><Clipboard size={15} /></button>
            </div>
            <div className="receipt-row">
              <span>Recipient</span>
              <strong title={receipt.recipient}>{shortId(receipt.recipient, 12, 10)}</strong>
              <button type="button" onClick={() => copy(receipt.recipient)} aria-label="Copy recipient address"><Clipboard size={15} /></button>
            </div>
            <div className="receipt-row">
              <span>Transaction</span>
              <strong title={receipt.transactionDigest}>{shortId(receipt.transactionDigest, 12, 10)}</strong>
              <button type="button" onClick={() => copy(receipt.transactionDigest)} aria-label="Copy transaction digest"><Clipboard size={15} /></button>
            </div>
            <div className="receipt-row">
              <span>Recorded</span>
              <strong>{new Date(receipt.recordedAt).toLocaleString()}</strong>
              <i />
            </div>
          </div>

          <div className="receipt-actions">
            <button type="button" className="ghost-button" onClick={() => copy(receipt.transactionDigest)}>
              <Clipboard size={16} /> Copy transaction ID
            </button>
            <a
              className="receipt-explorer"
              href={`https://suiscan.xyz/testnet/tx/${receipt.transactionDigest}`}
              target="_blank"
              rel="noreferrer"
            >
              Verify on Sui Explorer <ExternalLink size={16} />
            </a>
          </div>

          <p className="receipt-note">
            This receipt is generated only after the backend reports a successful live settlement.
            The transaction digest links to the public Sui Testnet record.
          </p>
        </section>
      ) : null}

      <section className="architecture-strip">
        <div><span>01</span><strong>Wallet</strong><small>Customer signs</small></div>
        <ArrowRight size={18} />
        <div><span>02</span><strong>Move escrow</strong><small>SUI locked</small></div>
        <ArrowRight size={18} />
        <div><span>03</span><strong>OpenCLIP</strong><small>Haircut scored</small></div>
        <ArrowRight size={18} />
        <div><span>04</span><strong>Oracle</strong><small>Score submitted</small></div>
        <ArrowRight size={18} />
        <div><span>05</span><strong>Settlement</strong><small>Pay or refund</small></div>
      </section>

      <footer>
        <div><Scissors size={16} /> Blocks & Scissors</div>
        <p>Hackathon Testnet build · AI similarity score is a demo heuristic, not a scientific probability.</p>
      </footer>
    </main>
  );
}
