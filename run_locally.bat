@echo off
echo ==========================================
echo   Fraud Risk Intelligence - Local Suite
echo ==========================================
echo.
echo [1/2] Starting Backend API (Uvicorn)...
start cmd /k "python -m uvicorn api:app --host 127.0.0.1 --port 8000"

echo [2/2] Starting Frontend Dashboard (Streamlit)...
start cmd /k "python -m streamlit run app.py --server.port 8501"

echo.
echo All systems launching! 
echo Keep the terminal windows open while using the app.
echo.
pause
