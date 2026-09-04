@echo off
title Blocks and Scissors

set "PROJECT=C:\Users\Rayyan Babar\Desktop\Blocks and Scissors"

echo ========================================
echo   Starting Blocks and Scissors
echo ========================================

echo.
echo Starting backend...
start "Blocks and Scissors - Backend" powershell.exe -NoExit -ExecutionPolicy Bypass -Command "Set-Location -LiteralPath '%PROJECT%\backend'; & '.\.venv\Scripts\python.exe' -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload"

timeout /t 3 /nobreak >nul

echo Starting frontend...
start "Blocks and Scissors - Frontend" powershell.exe -NoExit -ExecutionPolicy Bypass -Command "Set-Location -LiteralPath '%PROJECT%\frontend'; npm run dev -- --webpack"

timeout /t 4 /nobreak >nul

echo Opening website...
start "" "http://localhost:3000"

exit
