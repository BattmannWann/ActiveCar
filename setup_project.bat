@echo off
setlocal enabledelayedexpansion
title ActiveCar Environment Setup Wizard

:MENU
cls
echo ===================================================================
echo             ActiveCar / AssettoCorsaGym Setup Wizard
echo ===================================================================
echo.
echo Current Directory: %~dp0
echo Target Environment: p309
echo.
echo Please select an option:
echo.
echo   [1] Full Setup (Find paths, configure PYTHONPATH, install deps, verify)
echo   [2] Auto-Locate AssettoCorsaEnv ^& Configure PYTHONPATH in Conda
echo   [3] Install/Upgrade Core Python Dependencies (mss, pygetwindow, etc.)
echo   [4] Register Jupyter Kernel (Python p309)
echo   [5] Run Environment ^& AssettoCorsaEnv Import Diagnostic Check
echo   [6] Exit
echo.
echo ===================================================================
set /p "CHOICE=Enter choice [1-6]: "

if "%CHOICE%"=="1" goto FULL_SETUP
if "%CHOICE%"=="2" goto SETUP_PATHS
if "%CHOICE%"=="3" goto INSTALL_DEPS
if "%CHOICE%"=="4" goto SETUP_KERNEL
if "%CHOICE%"=="5" goto RUN_TEST
if "%CHOICE%"=="6" goto EXIT_SCRIPT
goto MENU

:: =====================================================================
:: 1. FULL SETUP
:: =====================================================================
:FULL_SETUP
cls
echo -------------------------------------------------------------------
echo  [1] FULL SETUP CONFIRMATION
echo -------------------------------------------------------------------
echo This will:
echo   - Activate Conda environment 'p309'
echo   - Auto-detect where 'AssettoCorsaEnv' is located on your disk
echo   - Permanently register all required paths into Conda's PYTHONPATH
echo   - Install missing dependencies (mss, pygetwindow, etc.)
echo   - Register the p309 Jupyter kernel
echo   - Run the diagnostic import check
echo.
set /p "CONFIRM=Do you want to proceed? (Y/N): "
if /i not "%CONFIRM%"=="Y" goto MENU

call :ACTIVATE_CONDA
if %ERRORLEVEL% neq 0 goto PAUSE_MENU

call :DO_SETUP_PATHS
call :DO_INSTALL_DEPS
call :DO_SETUP_KERNEL
call :DO_RUN_TEST
goto PAUSE_MENU

:: =====================================================================
:: 2. REGISTER PATHS
:: =====================================================================
:SETUP_PATHS
cls
echo -------------------------------------------------------------------
echo  [2] CONFIGURE PYTHONPATH IN CONDA
echo -------------------------------------------------------------------
echo This will scan for 'AssettoCorsaEnv' and set PYTHONPATH permanently
echo in the 'p309' Conda environment.
echo.
set /p "CONFIRM=Proceed with path search and configuration? (Y/N): "
if /i not "%CONFIRM%"=="Y" goto MENU

call :ACTIVATE_CONDA
if %ERRORLEVEL% neq 0 goto PAUSE_MENU
call :DO_SETUP_PATHS
goto PAUSE_MENU

:: =====================================================================
:: 3. INSTALL DEPENDENCIES
:: =====================================================================
:INSTALL_DEPS
cls
echo -------------------------------------------------------------------
echo  [3] INSTALL CORE PYTHON DEPENDENCIES
echo -------------------------------------------------------------------
echo Packages to install / upgrade:
echo   - mss, pygetwindow (Desktop frame grabber)
echo   - matplotlib, ipykernel, ipympl, ipywidgets
echo   - typing-extensions (Upgrade to prevent IPython TypeVar conflict)
echo.
set /p "CONFIRM=Proceed with package installation? (Y/N): "
if /i not "%CONFIRM%"=="Y" goto MENU

call :ACTIVATE_CONDA
if %ERRORLEVEL% neq 0 goto PAUSE_MENU
call :DO_INSTALL_DEPS
goto PAUSE_MENU

:: =====================================================================
:: 4. REGISTER JUPYTER KERNEL
:: =====================================================================
:SETUP_KERNEL
cls
echo -------------------------------------------------------------------
echo  [4] REGISTER JUPYTER KERNEL
echo -------------------------------------------------------------------
echo Links the active 'p309' environment to Jupyter as 'Python (p309)'.
echo.
set /p "CONFIRM=Register Jupyter kernel? (Y/N): "
if /i not "%CONFIRM%"=="Y" goto MENU

call :ACTIVATE_CONDA
if %ERRORLEVEL% neq 0 goto PAUSE_MENU
call :DO_SETUP_KERNEL
goto PAUSE_MENU

:: =====================================================================
:: 5. DIAGNOSTIC IMPORT CHECK
:: =====================================================================
:RUN_TEST
cls
echo -------------------------------------------------------------------
echo  [5] RUN DIAGNOSTIC IMPORT CHECK
echo -------------------------------------------------------------------
echo Testing imports in p309: torch, gym, cv2, mss, pygetwindow, AssettoCorsaEnv
echo.
call :ACTIVATE_CONDA
if %ERRORLEVEL% neq 0 goto PAUSE_MENU
call :DO_RUN_TEST
goto PAUSE_MENU

:: =====================================================================
:: SUBROUTINES
:: =====================================================================

:ACTIVATE_CONDA
echo [STEP] Checking Conda activation...
where conda >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] 'conda' command not found. Run this from an Anaconda Prompt!
    exit /b 1
)
echo [STEP] Activating environment 'p309'...
call conda activate p309
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to activate 'p309'.
    exit /b 1
)
echo [OK] 'p309' is active.
exit /b 0

:DO_SETUP_PATHS
echo.
echo [STEP] Searching for 'AssettoCorsaEnv' directory...

set "FOUND_AC_ENV="
set "SEARCH_BASE=%~dp0"
:: Check within current project tree
for /d /r "%SEARCH_BASE%" %%d in (*AssettoCorsaEnv) do (
    if exist "%%d\ac_client.py" (
        set "FOUND_AC_ENV=%%~dpd"
        goto :FOUND_PATH
    )
)

:: If not in ActiveCar, check parent directory (Masters Project)
for /d /r "%SEARCH_BASE%..\" %%d in (*AssettoCorsaEnv) do (
    if exist "%%d\ac_client.py" (
        set "FOUND_AC_ENV=%%~dpd"
        goto :FOUND_PATH
    )
)

:FOUND_PATH
if "!FOUND_AC_ENV!"=="" (
    echo [ERROR] Could not automatically find 'AssettoCorsaEnv' containing ac_client.py!
    set /p "FOUND_AC_ENV=Please manually paste the folder path containing AssettoCorsaEnv: "
)

:: Remove trailing backslash if present
if "!FOUND_AC_ENV:~-1!"=="\" set "FOUND_AC_ENV=!FOUND_AC_ENV:~0,-1!"
echo [OK] Located AssettoCorsaEnv parent at: !FOUND_AC_ENV!

set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

:: Build comprehensive PYTHONPATH string
set "NEW_PYPATH=%ROOT%;%ROOT%\src;!FOUND_AC_ENV!;!FOUND_AC_ENV!\assetto_corsa_gym;%ROOT%\algorithm\discor"

echo [STEP] Setting permanent Conda PYTHONPATH...
call conda env config vars set PYTHONPATH="!NEW_PYPATH!" -n p309
echo [OK] Successfully set Conda PYTHONPATH to:
echo     !NEW_PYPATH!

:: Also write a fallback .pth file directly into site-packages using Python itself
echo [STEP] Writing fallback .pth file into site-packages...
python -c "import site, os; p = os.path.join(site.getsitepackages()[-1], 'activecar_paths.pth'); f = open(p, 'w'); [f.write(x + '\n') for x in r'!NEW_PYPATH!'.split(';') if x]; f.close(); print('[OK] Fallback pth written to:', p)"

exit /b 0

:DO_INSTALL_DEPS
echo.
echo [STEP] Installing / Upgrading required packages...
call pip install --upgrade "typing-extensions>=4.8.0"
call pip install mss pygetwindow matplotlib ipykernel ipympl ipywidgets
echo [OK] Package setup complete.
exit /b 0

:DO_SETUP_KERNEL
echo.
echo [STEP] Registering Jupyter kernel 'p309'...
python -m ipykernel install --user --name p309 --display-name "Python (p309)"
if %ERRORLEVEL% equ 0 (
    echo [OK] Jupyter kernel successfully installed.
) else (
    echo [WARNING] Jupyter kernel installation failed.
)
exit /b 0

:DO_RUN_TEST
echo.
echo [STEP] Re-activating p309 so new Conda environment variables take effect...
call conda activate p309

echo [STEP] Executing verification test in Python...
python -c "import sys; print('\n[INFO] Active sys.path paths:'); [print('  ->', p) for p in sys.path if 'ActiveCar' in p or 'assetto' in p.lower() or 'p309' in p]; import torch; import gym; import cv2; import mss; import pygetwindow; from AssettoCorsaEnv.ac_client import Client; print('\n===================================================\n[SUCCESS] AssettoCorsaEnv and all modules loaded!\n===================================================')"
if %ERRORLEVEL% neq 0 (
    echo.
    echo [FAIL] Verification test encountered an error. Check if paths were registered above.
) else (
    echo.
    echo [OK] Verification test passed completely.
)
exit /b 0

:PAUSE_MENU
echo.
echo Press any key to return to the menu...
pause >nul
goto MENU

:EXIT_SCRIPT
echo Exiting setup wizard.
exit /b 0