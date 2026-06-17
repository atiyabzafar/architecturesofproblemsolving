@echo off
REM Double-click this file to launch the problem-solving model viewer.
REM It starts the local Python server and opens your browser automatically.
REM To stop it, just close this black window.
cd /d "%~dp0"
py -u serve.py
pause
