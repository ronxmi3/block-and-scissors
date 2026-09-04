import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Blocks & Scissors',
  description: 'AI-verified haircut escrow on Sui',
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
