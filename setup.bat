@echo off
REM CARELINK AI Microservice - Quick Setup Script for Windows

echo ==============================================
echo CARELINK AI Microservice - Setup
echo ==============================================
echo.

REM Check Python version
echo Checking Python version...
python --version

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv
call venv\Scripts\activate

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Download spaCy model
echo Downloading spaCy model...
python -m spacy download en_core_web_sm

REM Create .env file
if not exist .env (
    echo Creating .env file...
    copy .env.example .env
    echo .env file created. Please configure it before running.
)

REM Create directories
echo Creating directories...
if not exist logs mkdir logs
if not exist data mkdir data
if not exist app\models mkdir app\models

REM Generate synthetic data
echo Generating synthetic training data...
python data\synthetic_generator.py

REM Train model
echo Training ML model...
python training\train_model.py

echo.
echo ==============================================
echo Setup completed successfully!
echo ==============================================
echo.
echo To start the service:
echo   venv\Scripts\activate
echo   python -m uvicorn app.main:app --reload
echo.
echo Or use Docker:
echo   docker-compose up -d
echo.
echo API Documentation:
echo   http://localhost:8000/docs
echo.
pause
