@echo off
setlocal
cd /d "%~dp0..\.."
if errorlevel 1 exit /b 1
set "PYTHONPATH=%CD%\src\transformers\src;%CD%\src;%PYTHONPATH%"
if not defined NPROC_PER_NODE set "NPROC_PER_NODE=1"
if "%~1"=="" (
    torchrun --nproc_per_node=%NPROC_PER_NODE% src\train.py config\ad_bev_v4.yaml
) else (
    torchrun --nproc_per_node=%NPROC_PER_NODE% src\train.py %*
)
exit /b %errorlevel%
