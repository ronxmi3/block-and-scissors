@echo off
title Blocks and Scissors

set "PROJECT=C:\Users\Rayyan Babar\Desktop\Blocks and Scissors"

echo ========================================
echo       BLOCKS AND SCISSORS
echo ========================================

echo.
echo Starting backend on port 8000...

start "Blocks and Scissors - Backend" powershell.exe -NoExit -ExecutionPolicy Bypass -Command ^
"Set-Location -LiteralPath '%PROJECT%\backend'; & '.\.venv\Scripts\python.exe' -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload"

timeout /t 3 /nobreak >nul

echo Starting frontend on port 3010...

start "Blocks and Scissors - Frontend" powershell.exe -NoExit -ExecutionPolicy Bypass -Command ^
"Set-Location -LiteralPath '%PROJECT%\frontend'; npm run dev -- --webpack -p 3010"

timeout /t 5 /nobreak >nul

echo Opening Blocks and Scissors...

start "" "http://127.0.0.1:3010"

exit
