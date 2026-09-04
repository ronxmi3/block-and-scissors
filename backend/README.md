# Haircut Escrow — Phase 2 Backend

## What this adds

Phase 1 already proved:

- SUI can be locked in `HaircutEscrow`
- score >= threshold pays the barber
- score < threshold refunds the customer
- only the OracleCap can authorize settlement

Phase 2 replaces the manually typed score with:

1. Reference image upload
2. After image upload
3. OpenCLIP image embeddings
4. Hair-biased multi-crop similarity
5. 0–100 MVP score
6. Python invocation of the existing Sui `resolve_escrow` transaction

## Folder

Place this folder next to your Move package:

```text
Blocks and Scissors/
├── haircut_escrow/
└── backend/
```

## Install / run

```powershell
cd "C:\Users\Rayyan Babar\Desktop\Blocks and Scissors\backend"
.\run.ps1
```

Open:

```text
http://127.0.0.1:8000/docs
```

FastAPI gives you an interactive page where you can upload the two images.

## First AI-only test

```powershell
python test_score.py "C:\path\reference.jpg" "C:\path\after.jpg"
```

The first run downloads the OpenCLIP weights, so it is slower than later runs.

## API workflow

### `POST /score`

Uploads:

- `reference`
- `result`

Returns the similarity score without touching Sui.

### `POST /evaluate-and-resolve`

Uploads:

- `escrow_id`
- `reference`
- `result`
- `dry_run`

Header:

```text
X-Oracle-Key: value-from-your-.env
```

Keep `dry_run=true` until the returned Sui transaction succeeds.

Then use `dry_run=false` to commit the Testnet settlement.

## Important limitation

The current image crop is a heuristic. It is not true hair segmentation.

The score is also not a probability. Before demo day we should collect a small
set of "good match" and "bad match" pairs and tune the calibration values in
`.env`.

That is the next Phase 2 improvement.
