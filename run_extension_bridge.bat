@echo off
title PII Shield Extension Bridge Server
echo ========================================================
echo   PII Shield -- Local Browser Extension Bridge Server
echo ========================================================
echo Running at http://127.0.0.1:8765
echo.
cd /d "%~dp0"
python -m extension_server.server
pause
