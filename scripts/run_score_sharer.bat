@echo off
chcp 65001 > nul
set PYTHONIOENCODING=utf-8
cd /d "C:\Users\user\OneDrive\Documents\GitHub\LinkedIn-Queens-Solver"
if not exist "%CD%\logs" mkdir "%CD%\logs"

echo ===== START %DATE% %TIME% ===== >> "%CD%\logs\score_sharer.log"

"C:\Users\user\.local\bin\uv.exe" run python "Queens\score_sharer.py" >> "%CD%\logs\score_sharer.log" 2>&1

echo ===== END %DATE% %TIME% ===== >> "%CD%\logs\score_sharer.log"

exit /b %ERRORLEVEL%