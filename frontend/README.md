# Blocks & Scissors — Phase 3 Frontend

A Next.js frontend for the Sui Testnet haircut escrow you built in Phases 1–2.

## What this frontend does

- Connects a browser Sui wallet on **Testnet** using the current Mysten dApp Kit.
- Creates a fresh `HaircutEscrow` from the UI and locks the selected SUI payment.
- Automatically reads the new shared escrow object ID from the creation transaction.
- Uploads reference + finished haircut images.
- Calls a **server-side Next.js proxy** which forwards the images to your Python oracle.
- Keeps `ORACLE_API_KEY` server-side so it is never exposed to browser JavaScript.
- Supports **Simulation mode** (dry run) and live Testnet settlement.
- Shows score, pay/refund verdict and Sui transaction digest.

## 1. Put it in your project

Recommended folder:

```text
C:\Users\Rayyan Babar\Desktop\Blocks and Scissors\frontend
```

## 2. Create `.env.local`

In this frontend folder:

```powershell
Copy-Item .env.local.example .env.local
notepad .env.local
```

Replace only:

```env
ORACLE_API_KEY=PASTE_YOUR_CURRENT_ORACLE_KEY_HERE
```

with the **same current secret** in `backend\.env`.

Important: keep the name exactly `ORACLE_API_KEY`. Do **not** rename it to `NEXT_PUBLIC_ORACLE_API_KEY`; that would leak it to the browser.

## 3. Keep the Phase 2 backend running

Terminal 1:

```powershell
cd "C:\Users\Rayyan Babar\Desktop\Blocks and Scissors\backend"
.\run.ps1
```

You should still have FastAPI on:

```text
http://127.0.0.1:8000
```

## 4. Install + run the frontend

Terminal 2:

```powershell
cd "C:\Users\Rayyan Babar\Desktop\Blocks and Scissors\frontend"
npm install
npm run dev
```

Then open:

```text
http://localhost:3000
```

## 5. Connect a Testnet wallet

The website needs a browser Sui wallet to sign `create_escrow` as the customer.

The wallet must:

- be on **Sui Testnet**;
- contain Testnet SUI for the payment + gas.

The default barber address and package ID are already your live Phase 1 values.

## Normal app flow

```text
Connect wallet
   ↓
Choose barber + amount + threshold
   ↓
Lock payment in Move escrow
   ↓
Upload reference + finished haircut
   ↓
AI score from Python backend
   ↓
Oracle calls resolve_escrow
   ↓
score >= threshold → barber paid
score <  threshold → customer refunded
```

## Why the API key is not in the browser

The browser calls:

```text
POST /api/verify
```

The Next.js server route then attaches `X-Oracle-Key` and calls:

```text
http://127.0.0.1:8000/evaluate-and-resolve
```

So users never receive your oracle secret.

## Current MVP limitations

This frontend is intended for the current local/Testnet hackathon build. Before public deployment, the verify route should require wallet-signed authorization (or your later zkLogin/session flow), not merely an escrow ID. Otherwise a public caller could ask the oracle to evaluate an escrow.

The published Move contract does **not** yet commit a hash of the reference image when escrow is created. For a stronger version, the next contract upgrade should bind the agreed reference image (or a content hash) to the escrow so it cannot be swapped later.

## Test safely

Leave **Simulation mode ON** for the first frontend test. Once the AI score and Sui dry run both look correct, turn it OFF to execute the real Testnet settlement.
