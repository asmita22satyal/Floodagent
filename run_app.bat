@echo off
REM Starts FloodAgent. Uses the virtual environment created at C:\FloodAgent_work\venv
REM To use Gemini: set GEMINI_API_KEY=your_key   (do not save the key in any file)
cd /d "%~dp0"
"C:\FloodAgent_work\venv\Scripts\python.exe" -I -m streamlit run app.py
