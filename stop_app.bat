@echo off
title Stop Blocks and Scissors

echo Stopping Blocks and Scissors...

powershell.exe -NoProfile -Command "$ports = 8000,3000,3001; foreach ($port in $ports) { $connections = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue; foreach ($connection in $connections) { if ($connection.OwningProcess -gt 0) { Stop-Process -Id $connection.OwningProcess -Force -ErrorAction SilentlyContinue } } }"

echo.
echo Blocks and Scissors stopped.
timeout /t 2 /nobreak >nul
exit
