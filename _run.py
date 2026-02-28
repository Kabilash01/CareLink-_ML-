"""Launcher: ensures cwd is ai-service before starting uvicorn."""
import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
import uvicorn
uvicorn.run("app.main:app", host="0.0.0.0", port=8000)
