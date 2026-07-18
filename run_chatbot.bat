@echo off
REM Force working directory to this script's own folder (Windows resets
REM cwd to System32 when a .bat is run elevated).
cd /d %~dp0

REM oneAPI's setvars.bat sets up icx.exe, SYCL, and Level Zero linkage,
REM which the Intel Triton XPU backend requires (plain MSVC cl.exe can't
REM compile the SYCL code it generates). This is only needed for XPU -
REM chatbot.py itself now falls back to CUDA or CPU automatically if no
REM XPU is found, so on a CUDA/CPU-only machine we just skip this whole
REM block instead of failing.
set ONEAPI_SETVARS="C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
if exist %ONEAPI_SETVARS% (
    REM setvars.bat looks for VS at standard year-named paths; this machine's
    REM VS is under a nonstandard "18" folder, so point it there explicitly.
    set "VS2026INSTALLDIR=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools"
    call %ONEAPI_SETVARS%

    set ICX="C:\Program Files (x86)\Intel\oneAPI\compiler\latest\bin\icx.exe"
    set CC=%ICX%
    set CXX=%ICX%

    REM Triton's compiled .pyd needs vcruntime140.dll/msvcp140.dll at runtime.
    set "VC_REDIST=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Redist\MSVC\14.51.36231\x64\Microsoft.VC145.CRT"
    set "PATH=%VC_REDIST%;%PATH%"

    echo [INFO] oneAPI environment initialized ^(XPU available^).
) else (
    echo [INFO] oneAPI not found - skipping XPU setup, chatbot.py will use CUDA or CPU instead.
)

set PYTHON="C:\Users\Diska\AppData\Local\Programs\Python\Python311\python.exe"
if not exist %PYTHON% (
    echo [ERROR] Could not find python.exe at %PYTHON%
    pause
    exit /b 1
)

%PYTHON% chatbot.py
pause
