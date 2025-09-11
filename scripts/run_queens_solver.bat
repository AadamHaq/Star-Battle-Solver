@echo off
cd /d "C:\Users\user\OneDrive\Documents\GitHub\Star-Battle-Solver"
if not exist "%CD%\logs" mkdir "%CD%\logs"

echo ===== START %DATE% %TIME% ===== >> "%CD%\logs\queens_solver.log"

"C:\Users\user\anaconda3\Scripts\conda.exe" run -n star_battle --no-capture-output python "Queens\queens_solver.py" >> "%CD%\logs\queens_solver.log" 2>&1

echo ===== END %DATE% %TIME% ===== >> "%CD%\logs\queens_solver.log"

exit /b %ERRORLEVEL%
