# Blocks & Scissors

Blocks & Scissors is a small experiment I built around a pretty simple question:

**What happens if a customer and a barber disagree about whether a haircut actually matches the reference photo?**

The idea is to lock the payment in escrow before the haircut happens, compare the finished haircut against the reference using AI, and then let a smart contract handle the payment based on the result.

This project was built for the MUBA Blockchain Hackathon 2026, Sui Track.

---

## How it works

The flow is:

1. The customer connects a Sui wallet.
2. They enter the barber's address and the payment amount.
3. The payment is locked in a Sui escrow object.
4. A reference haircut image is uploaded.
5. After the haircut, the result image is uploaded.
6. The backend compares the two images and produces a similarity score from 0–100.
7. The score is submitted to the Sui smart contract by the backend oracle.
8. If the score is above the agreed threshold, the barber gets paid.
9. If the score is below the threshold, the customer gets refunded.

The current default threshold is:

```text
80
```

The threshold itself is stored in the escrow when it is created.

---

## Why I used Sui

The blockchain part is not just there to store the AI score.

The main reason for using Sui is the escrow.

Once the customer creates the escrow, the money is held by the smart contract instead of either the customer or the barber.

The AI only provides the score.

The Move contract decides what happens to the money.

So the basic idea is:

```text
AI decides
    ↓
Sui enforces
```

---

## Current architecture

```text
Next.js frontend
      |
      | HTTP
      v
FastAPI backend
      |
      |---- OpenCLIP haircut scoring
      |
      |---- Sui oracle transaction
                  |
                  v
            Move contract
                  |
             payout/refund
```

The frontend handles:

* wallet connection
* escrow creation
* image upload
* displaying AI results
* settlement receipts
* transaction verification

The backend handles:

* image validation
* AI inference
* haircut scoring
* oracle authorization
* submitting the final score to Sui

The Move contract handles:

* storing the customer and barber
* holding the funds
* storing the threshold
* storing the final score
* paying the barber
* refunding the customer
* preventing the escrow from being resolved twice

---

## AI scoring

The AI side uses OpenCLIP through PyTorch.

I originally started with basic image similarity, but that was not enough because two haircut photos can have different backgrounds, lighting, poses, clothing, etc.

The current scorer also looks at haircut-specific information such as:

* overall haircut family
* shape
* texture
* back length
* taper / fade structure
* fade height
* side coverage
* blending
* confidence of the classification

I also added a few things to make the score more stable:

* hair-focused image crops
* grayscale comparison
* blurred shape comparison
* horizontal image flipping for left/right profile photos
* multiple text prompts for fade classifications
* confidence-aware penalties
* haircut-family mismatch rules
* score caps for very different hairstyles

The current version is:

```text
phase2i-confidence-aware-fade-fusion
```

This is still an MVP model.

It works well enough to demonstrate clear matches and mismatches, but it is not meant to be treated as a production-grade haircut judge yet.

A real version would need a much larger labelled dataset and more controlled image capture.

---

## Smart contract

The Move contract contains two main objects.

### HaircutEscrow

The escrow stores:

```text
customer
barber
funds
threshold
score
status
```

The status values are:

```text
0 = Pending
1 = Barber Paid
2 = Customer Refunded
```

The settlement rule is basically:

```text
if score >= threshold
    pay barber
else
    refund customer
```

Once an escrow has been resolved, it cannot be resolved again.

---

### OracleCap

The backend cannot just resolve any escrow without permission.

The deployed contract uses an `OracleCap` object.

Only the holder of that capability can submit a score that resolves the escrow.

For the hackathon MVP, my backend acts as the trusted oracle.

---

## Sui testnet deployment

Network:

```text
Sui Testnet
```

Package ID:

```text
0x5d7fa930a5d95ae7f8a2a56693e5341cb8e84dd0865fa42a6aac215c2659057a
```

Module:

```text
haircut_escrow::haircut_escrow
```

Both paths have been tested on testnet:

```text
Barber payout
Customer refund
```

The frontend also displays the transaction digest after settlement so the transaction can be checked on a Sui explorer.

---

## Tech stack

```text
Frontend
Next.js
React
TypeScript

Backend
Python
FastAPI

AI
OpenCLIP
PyTorch
torchvision
Pillow

Blockchain
Sui
Move
Sui CLI
```

---

## Project structure

```text
Blocks and Scissors/
│
├── haircut_escrow/
│   ├── Move.toml
│   └── sources/
│       └── haircut_escrow.move
│
├── backend/
│   ├── ai/
│   │   └── scorer.py
│   │
│   ├── blockchain/
│   │   └── sui_oracle.py
│   │
│   ├── config.py
│   ├── main.py
│   ├── requirements.txt
│   └── .env.example
│
├── frontend/
│   ├── app/
│   ├── components/
│   └── package.json
│
└── README.md
```

---

# Running the project

## Requirements

You will need:

* Python
* Node.js
* npm
* Sui CLI
* a Sui testnet wallet
* testnet SUI

Check Sui:

```powershell
sui --version
sui client active-env
sui client active-address
```

The active environment should be:

```text
testnet
```

---

## Backend

Open a terminal:

```powershell
cd "C:\path\to\Blocks and Scissors\backend"
```

Create a virtual environment if needed:

```powershell
python -m venv .venv
```

Install the requirements:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Copy the example environment file:

```powershell
Copy-Item .env.example .env
```

Fill in the required local configuration.

Do not commit `.env`.

Start the backend:

```powershell
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

Backend:

```text
http://127.0.0.1:8000
```

Health check:

```text
http://127.0.0.1:8000/health
```

FastAPI docs:

```text
http://127.0.0.1:8000/docs
```

---

## Frontend

Open another terminal:

```powershell
cd "C:\path\to\Blocks and Scissors\frontend"
```

Install dependencies:

```powershell
npm install
```

Run the frontend:

```powershell
npm run dev -- --webpack
```

Open:

```text
http://localhost:3000
```

---

## Demo flow

For the demo I normally use two cases.

### Successful haircut

```text
Reference haircut
        +
Similar result
        ↓
AI score above Threshold(80)
        ↓
Barber paid
```

### Failed haircut

```text
Reference haircut
        +
Clearly different result
        ↓
AI score below Threshold (80)
        ↓
Customer refunded
```

After the escrow is settled, the frontend shows the transaction ID and a link to verify it on-chain.

---

## Current limitations

This is a hackathon MVP, so there are still quite a few things I would change before using it with real money.

The biggest ones are:

* the AI model needs proper calibration on a labelled haircut dataset
* the current workflow mainly uses side-profile images
* the reference image is not currently committed to the smart contract as a hash
* the backend oracle is centralized
* there is no timeout refund yet if the service is never completed
* customers currently need a normal Sui wallet
* there is no fiat/stablecoin off-ramp for barbers
* production wallets and keys would need much stronger security

---

## What I would add next

The next features I would work on are:

* front + side + back haircut verification
* storing a reference image hash in the escrow
* larger labelled haircut dataset
* a dispute/manual review range for uncertain scores
* zkLogin
* sponsored transactions
* timeout-based refunds
* stablecoin or fiat payment abstraction
* multiple independent AI/oracle services
* barber dashboard

---

## AI usage

AI was used both inside the product and during development.

Inside the application:

* OpenCLIP is used for image embeddings and similarity.
* PyTorch is used to run the model.
* Custom Python logic handles the haircut-specific scoring.

During development:

* ChatGPT was used for debugging, front end, testing help and documentation assistance.

The final application flow, contract behaviour and transactions were tested using the actual running project on Sui testnet.

---

## Security

Do not commit:

```text
.env
.env.local
private keys
recovery phrases
oracle API keys
```

This project is currently testnet only.

---

## Demo video

Add demo link here: https://youtu.be/uM_TnW5ubX0

```text
TODO
```

---

## Team

```text
Rayyan Babar
Hadia Bashir 
Badr 
```

---

## Final idea

Blocks & Scissors is basically an experiment in combining AI with programmable payments.

The haircut is just the easiest example to demonstrate visually.

The same pattern could potentially be used for other jobs where the final result can be compared against something that was agreed beforehand, such as tailoring, detailing, tattoo work, repairs or commissioned physical work.

**AI decides. Sui enforces.**
