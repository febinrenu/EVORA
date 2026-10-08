@echo off
rem Start evora on Windows.
rem
rem   start.bat           install what is missing, build the UI if it is stale,
rem                       then run the API with the UI on http://127.0.0.1:8700
rem   start.bat dev       API with fixture data (evora_MOCK=1) plus the UI dev
rem                       server with hot reload on http://localhost:5173
rem   start.bat rebuild   force a fresh UI build before starting
rem   start.bat setup     one-time: also install the perception stack (torch,
rem                       detector, embeddings) and download model weights
rem
rem Needs Node.js 20+ and Python on PATH. uv is found where its installers put
rem it, or installed on first run (pip of the real Python, else the official
rem installer); it fetches the Python 3.12 the backend needs by itself. Works
rem with or without an activated virtual environment.
setlocal EnableExtensions
cd /d "%~dp0"
set "MODE=%~1"

where node >nul 2>nul
if errorlevel 1 (
  echo Node.js 20 or newer is required: https://nodejs.org
  exit /b 1
)
call :find_uv
if errorlevel 1 exit /b 1

if not exist ".env" if exist ".env.example" (
  copy ".env.example" ".env" >nul
  echo Created .env from .env.example. Put your GROQ_KEYS there, or run on-prem with Ollama.
)

echo.
echo [1/4] Backend environment
if /i "%MODE%"=="setup" (
  echo       Installing the perception stack, this can take several minutes...
  %UV% --directory backend sync --extra perception --extra embed
  if errorlevel 1 goto :fail
  set "EVORA_MODELS_DIR=..\models"
  %UV% --directory backend run python ../scripts/models_download.py
  if errorlevel 1 echo       Some model downloads failed; the app still starts and says what is missing.
) else (
  rem --inexact keeps extras that an earlier "start.bat setup" installed
  %UV% --directory backend sync --quiet --inexact
  if errorlevel 1 goto :fail
)

echo [2/4] Frontend dependencies
if not exist "frontend\node_modules" (
  pushd frontend
  call npm ci --no-audit --no-fund
  if errorlevel 1 ( popd & goto :fail )
  popd
)

if /i "%MODE%"=="dev" goto :dev

echo [3/4] UI build
if not exist "frontend\public\footage\atlas.mp4" (
  where ffmpeg >nul 2>nul
  if not errorlevel 1 (
    echo       Building camera footage from the EPFL sequences, about a minute...
    pushd frontend
    call npm run footage
    popd
  ) else (
    echo       ffmpeg not found: the camera wall uses procedural feeds.
  )
)
rem rebuild when the UI sources are newer than the last build
set "NEED="
if /i "%MODE%"=="rebuild" set "NEED=1"
if not exist "frontend\dist\index.html" set "NEED=1"
if not defined NEED (
  powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\ui_stale.ps1"
  if errorlevel 1 set "NEED=1"
)
if defined NEED (
  pushd frontend
  call npm run build
  if errorlevel 1 ( popd & goto :fail )
  popd
) else (
  echo       UI build is up to date.
)

echo [4/4] Starting evora
set "EVORA_MODELS_DIR=..\models"
%UV% --directory backend run python ../scripts/up.py --open
exit /b %errorlevel%

:dev
echo [3/3] Dev mode: API with fixture data on :8700, UI with hot reload on :5173
start "evora API (fixtures)" cmd /k "set evora_MOCK=1&& %UV% --directory backend run uvicorn evora.api.app:create_app --factory --reload --host 127.0.0.1 --port 8700"
start "" cmd /c "timeout /t 8 /nobreak >nul & start http://localhost:5173/app/"
pushd frontend
call npm run dev
popd
exit /b 0

:find_uv
rem An activated virtual environment (often backend\.venv, which has no pip)
rem must not decide which Python is used; uv picks backend\.venv by itself.
set "VIRTUAL_ENV="
where uv >nul 2>nul
if not errorlevel 1 ( set "UV=uv" & exit /b 0 )
rem where the uv installer, cargo, winget and "pip install --user" put it
for %%P in ("%USERPROFILE%\.local\bin" "%USERPROFILE%\.cargo\bin" "%LOCALAPPDATA%\Microsoft\WinGet\Links") do (
  if exist "%%~P\uv.exe" ( set "PATH=%%~P;%PATH%" & set "UV=uv" & exit /b 0 )
)
for /d %%D in ("%APPDATA%\Python\Python3*") do (
  if exist "%%~D\Scripts\uv.exe" ( set "PATH=%%~D\Scripts;%PATH%" & set "UV=uv" & exit /b 0 )
)
echo Installing uv (Python package manager)...
call :base_python
if not defined BASEPY goto :uv_installer
"%BASEPY%" -m pip --version >nul 2>nul
if errorlevel 1 goto :uv_installer
"%BASEPY%" -m pip install --user --quiet uv
if errorlevel 1 goto :uv_installer
set "UVDIR="
for /f "usebackq delims=" %%S in (`call "%BASEPY%" -c "import os, sysconfig; print(sysconfig.get_path('scripts', os.name + '_user'))"`) do set "UVDIR=%%S"
if defined UVDIR if exist "%UVDIR%\uv.exe" ( set "PATH=%UVDIR%;%PATH%" & set "UV=uv" & exit /b 0 )
:uv_installer
echo       pip could not install it here; trying the official uv installer...
powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
if exist "%USERPROFILE%\.local\bin\uv.exe" ( set "PATH=%USERPROFILE%\.local\bin;%PATH%" & set "UV=uv" & exit /b 0 )
echo Could not install uv. Install it from https://docs.astral.sh/uv/ and run start.bat again.
exit /b 1

:base_python
rem the real Python, not a virtual environment's copy: the py launcher first,
rem then whatever "python" is, asked for the interpreter its venv was made from
set "BASEPY="
for /f "usebackq delims=" %%E in (`py -3 -c "import sys; print(sys.executable)" 2^>nul`) do set "BASEPY=%%E"
if defined BASEPY exit /b 0
for /f "usebackq delims=" %%E in (`python -c "import sys; print(getattr(sys, '_base_executable', sys.executable))" 2^>nul`) do set "BASEPY=%%E"
exit /b 0

:fail
echo.
echo evora did not start. Read the message above; "make doctor" lists what is missing.
exit /b 1
