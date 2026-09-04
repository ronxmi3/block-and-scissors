'use client';

import { DAppKitProvider } from '@mysten/dapp-kit-react';
import { dAppKit } from '@/lib/dapp-kit';
import { HaircutDashboard } from './HaircutDashboard';

export default function WalletApp() {
  return (
    <DAppKitProvider dAppKit={dAppKit}>
      <HaircutDashboard />
    </DAppKitProvider>
  );
}
