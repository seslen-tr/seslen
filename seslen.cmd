@echo off
rem Seslen: http://127.0.0.1:8780 (yalnız bu bilgisayar, internetsiz, GPU gerekmez)
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
start "" /b cmd /c "timeout /t 25 >nul && start http://127.0.0.1:8780"
.venv\Scripts\python.exe -u ogrenci\studyoSunucu.py
