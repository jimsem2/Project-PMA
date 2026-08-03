@echo off
rem ---------------------------------------------------------------------------
rem Launch the drag-and-drop DAT -> TIF converter.
rem
rem Double-click this file, or drag .dat files / folders straight onto it.
rem
rem It looks for a Python interpreter in this order:
rem   1) the project's own virtual environment (..\.venv), if present
rem   2) pythonw.exe on the system PATH
rem   3) the Windows "py" launcher
rem so it works both inside the full project and when the tools\ folder has
rem been copied to another PC on its own.
rem ---------------------------------------------------------------------------
setlocal
set "HERE=%~dp0"
set "GUI=%HERE%dat_converter_gui.py"
set "VENVW=%HERE%..\.venv\Scripts\pythonw.exe"

if exist "%VENVW%" (
    start "" "%VENVW%" "%GUI%" %*
    exit /b 0
)

where pythonw.exe >nul 2>&1
if not errorlevel 1 (
    start "" pythonw.exe "%GUI%" %*
    exit /b 0
)

where pyw.exe >nul 2>&1
if not errorlevel 1 (
    start "" pyw.exe -3 "%GUI%" %*
    exit /b 0
)

echo.
echo Could not find a Python interpreter.
echo.
echo Install Python 3.12 from https://www.python.org/downloads/
echo (tick "Add python.exe to PATH" during setup), then install the few
echo packages this tool needs:
echo.
echo     python -m pip install -r "%HERE%requirements-tools.txt"
echo.
echo If the window opens but conversion fails, run it with a console to see
echo the error message:
echo.
echo     python "%GUI%"
echo.
pause
exit /b 1
