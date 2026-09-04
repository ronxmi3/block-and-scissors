'use client';

import dynamic from 'next/dynamic';

const WalletApp = dynamic(() => import('@/components/WalletApp'), {
  ssr: false,
  loading: () => (
    <main className="loading-screen">
      <div className="loading-mark">B/S</div>
      <p>Loading Sui wallet layer…</p>
    </main>
  ),
});

export default function Home() {
  return <WalletApp />;
}
