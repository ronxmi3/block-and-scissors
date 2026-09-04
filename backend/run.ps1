# Run this from inside the backend folder.

# Create the environment if it does not exist.
if (-not (Test-Path ".venv")) {
    python -m venv .venv
}

# Activate it.
.\.venv\Scripts\Activate.ps1

# Upgrade pip and install dependencies.
python -m pip install --upgrade pip
pip install -r requirements.txt

# Create .env from the template on first run.
if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host ""
    Write-Host "Created .env from .env.example."
    Write-Host "IMPORTANT: edit ORACLE_API_KEY before exposing the backend."
    Write-Host ""
}

# Start FastAPI.
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
