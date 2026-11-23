@echo off
chcp 65001 >nul
cd /d "%~dp0.."
echo ==========================================
echo   拍照识垃圾 · YOLOv8 上传识别
echo ==========================================
echo 正在启动服务，请稍候...
start "" http://127.0.0.1:5000
"%~dp0..\.venv\Scripts\python.exe" "%~dp0app.py"
pause
