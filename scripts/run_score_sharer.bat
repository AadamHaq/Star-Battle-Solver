@echo off
REM --- go to repo root ---
cd /d "C:\Users\user\OneDrive\Documents\GitHub\Star-Battle-Solver"

REM --- ensure logs folder exists ---
if not exist "%CD%\logs" mkdir "%CD%\logs"

REM --- timestamp start ---
echo ===== START %DATE% %TIME% ===== >> "%CD%\logs\score_sharer.log"

REM --- run inside conda env (recommended for Task Scheduler) ---
"C:\Users\user\anaconda3\Scripts\conda.exe" run -n star_battle --no-capture-output python "Queens\score_sharer.py" >> "%CD%\logs\score_sharer.log" 2>&1

REM --- timestamp end ---
echo ===== END %DATE% %TIME% ===== >> "%CD%\logs\score_sharer.log"

exit /b %ERRORLEVEL%
