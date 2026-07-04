"""
Dashboard Runner for Temporal-Causal GraphRAG Next-TTP Prediction

Orchestrates and launches both the FastAPI backend and the React Vite frontend.
"""
import os
import sys
import subprocess
import time
import webbrowser
import signal

base_dir = os.path.dirname(os.path.abspath(__file__))

def get_python_exec():
    # Use virtual environment python if available
    venv_py = os.path.join(base_dir, ".venv", "Scripts", "python.exe" if os.name == 'nt' else "bin/python")
    if os.path.exists(venv_py):
        return venv_py
    return sys.executable

def main():
    print("===============================================================")
    print(" STARTING TEMPORAL-CAUSAL GRAPHRAG NEXT-TTP PREDICTION DASHBOARD ")
    print("===============================================================")
    
    python_exec = get_python_exec()
    print(f"Using Python executable: {python_exec}")

    # 1. Start FastAPI Backend
    print("\n[1/3] Launching FastAPI Backend on http://127.0.0.1:8000 ...")
    backend_proc = subprocess.Popen(
        [python_exec, "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", "8000"],
        cwd=base_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    # Simple check to see if backend starts successfully
    time.sleep(2.0)
    if backend_proc.poll() is not None:
        print("FAIL: Backend failed to start. Logs:")
        print(backend_proc.stdout.read())
        sys.exit(1)
    print("Backend started successfully.")

    # 2. Start Vite Frontend
    print("\n[2/3] Launching React Vite Frontend on http://localhost:5173 ...")
    npm_cmd = "npm.cmd" if os.name == 'nt' else "npm"
    frontend_proc = subprocess.Popen(
        [npm_cmd, "run", "dev"],
        cwd=os.path.join(base_dir, "frontend"),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    time.sleep(2.0)
    if frontend_proc.poll() is not None:
        print("FAIL: Frontend failed to start. Logs:")
        print(frontend_proc.stdout.read())
        backend_proc.terminate()
        sys.exit(1)
    print("Frontend started successfully.")

    # 3. Open Web Browser
    print("\n[3/3] Launching dashboard in your default browser...")
    webbrowser.open("http://localhost:5173")

    print("\nDashboard is running! Press Ctrl+C to terminate both servers.")
    
    try:
        # Keep main thread alive and print logs if any errors occur
        while True:
            # Check backend status
            if backend_proc.poll() is not None:
                print("Backend terminated unexpectedly.")
                break
            # Check frontend status
            if frontend_proc.poll() is not None:
                print("Frontend terminated unexpectedly.")
                break
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nTerminating servers...")
    finally:
        # Graceful cleanup
        backend_proc.terminate()
        frontend_proc.terminate()
        print("Servers successfully stopped.")

if __name__ == "__main__":
    main()
