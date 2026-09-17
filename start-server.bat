@echo off
setlocal

rem ---------------------------------------------------------------------------
rem Configuracion: ajusta estas rutas y valores antes de ejecutar el archivo.
rem ---------------------------------------------------------------------------
set "LLAMA_DIR=C:\llama.cpp"
set "MODEL_PATH=C:\llama.cpp\models\model.gguf"
set "HOST=127.0.0.1"
set "PORT=8080"
set "CTX_SIZE=8192"

rem Comprueba que los archivos principales existen antes de arrancar.
if not exist "%LLAMA_DIR%\llama-server.exe" (
  echo [ERROR] No se encuentra "%LLAMA_DIR%\llama-server.exe".
  echo Revisa la variable LLAMA_DIR.
  pause
  exit /b 1
)

if not exist "%MODEL_PATH%" (
  echo [ERROR] No se encuentra el modelo "%MODEL_PATH%".
  echo Revisa la variable MODEL_PATH.
  pause
  exit /b 1
)

cd /d "%LLAMA_DIR%"
if errorlevel 1 (
  echo [ERROR] No se pudo abrir la carpeta "%LLAMA_DIR%".
  pause
  exit /b 1
)

echo Iniciando llama-server en http://%HOST%:%PORT%
echo Pulsa Ctrl+C para detenerlo.
echo.

llama-server.exe ^
  --model "%MODEL_PATH%" ^
  --n-gpu-layers all ^
  --ctx-size %CTX_SIZE% ^
  --flash-attn on ^
  --host %HOST% ^
  --port %PORT%

set "SERVER_EXIT_CODE=%ERRORLEVEL%"
echo.
echo llama-server ha terminado con el codigo %SERVER_EXIT_CODE%.
pause
exit /b %SERVER_EXIT_CODE%
