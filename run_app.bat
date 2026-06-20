@echo off

echo Starting PBU Analyzer...

start "PBU Analyzer" cmd /k python app.py

timeout /t 3 /nobreak > nul

start "" http://127.0.0.1:5000

exit