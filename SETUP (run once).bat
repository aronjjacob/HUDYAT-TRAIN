@echo off
setlocal
cd /d "%~dp0"
title HUDYAT Trainer - setup
echo ==============================================================
echo   HUDYAT Trainer - setup (run once per laptop)
echo   This takes 5-15 minutes the first time (TensorFlow is large).
echo ==============================================================
echo.

where py >nul 2>nul
if errorlevel 1 (
  echo Python is not installed.
  echo Install "Python 3.12 (64-bit)" from https://www.python.org/downloads/windows/
  echo and run this file again.
  goto :fail
)
py -3.12 -c "import sys; sys.exit(0 if sys.maxsize > 2**32 else 1)" >nul 2>nul
if errorlevel 1 (
  echo Python 3.12 64-bit was not found.
  echo Install "Python 3.12 (64-bit)" from https://www.python.org/downloads/windows/
  echo and run this file again.
  goto :fail
)

if not exist "venv\Scripts\python.exe" (
  echo [1/4] Creating the Python environment in this folder...
  py -3.12 -m venv venv
  if errorlevel 1 goto :fail
) else (
  echo [1/4] Python environment already exists.
)

echo [2/4] Installing the packages (requirements.txt)...
"venv\Scripts\python.exe" -m pip install --upgrade pip --quiet --disable-pip-version-check
"venv\Scripts\python.exe" -m pip install -r requirements.txt --quiet --disable-pip-version-check
if errorlevel 1 (
  echo.
  echo Installing the packages failed. Check the internet connection and run this file again.
  echo If the message above mentions "long path", the folder is too deep: move it closer to C:\
  echo ^(for example C:\HUDYAT - TRAINER^), delete the venv folder inside it, and run this file again.
  goto :fail
)

echo [3/4] Checking the MediaPipe model and the packages...
"venv\Scripts\python.exe" tools\setup_check.py
if errorlevel 1 goto :fail

echo [4/4] Test tools (only needed by the developers' tests; failures here are not a problem)...
"venv\Scripts\python.exe" -m playwright install chromium >nul 2>nul
where npm >nul 2>nul
if not errorlevel 1 if not exist "tools\tfjs_check\node_modules\@tensorflow\tfjs" (
  pushd tools\tfjs_check
  call npm install --silent >nul 2>nul
  popd
)

echo.
echo ==============================================================
echo   Setup finished. Start the app with "START HUDYAT TRAINER.bat".
echo ==============================================================
if not "%~1"=="--no-pause" pause
exit /b 0

:fail
echo.
echo Setup did NOT finish. Read the message above, fix it, and run this file again.
if not "%~1"=="--no-pause" pause
exit /b 1
