@echo off
setlocal
cd /d "%~dp0"
set "URL=http://localhost:8517"

if not exist "venv\Scripts\python.exe" (
  echo The app is not set up yet. Double-click "SETUP (run once).bat" first.
  pause
  exit /b 1
)

rem Already running? Just open the browser.
powershell -NoProfile -Command "try { if ((Invoke-WebRequest -UseBasicParsing '%URL%/_stcore/health' -TimeoutSec 2).Content -eq 'ok') { exit 0 } } catch {} ; exit 1" >nul 2>nul
if not errorlevel 1 (
  start "" "%URL%"
  exit /b 0
)

title HUDYAT Trainer - keep this window open
echo HUDYAT Trainer is starting. The browser opens in a few seconds.
echo Keep this window open while you use the app. Close it to stop the app.
echo.
rem Open the browser as soon as the app answers.
start "" /b powershell -NoProfile -WindowStyle Hidden -Command "for ($i = 0; $i -lt 240; $i++) { try { if ((Invoke-WebRequest -UseBasicParsing '%URL%/_stcore/health' -TimeoutSec 1).Content -eq 'ok') { Start-Process '%URL%'; break } } catch {} ; Start-Sleep -Milliseconds 500 }"
"venv\Scripts\python.exe" -m streamlit run trainer_app.py --server.headless true --server.port 8517
if errorlevel 1 (
  echo.
  echo The app stopped with an error ^(see above^). Run "SETUP (run once).bat" again, then try again.
  pause
)
