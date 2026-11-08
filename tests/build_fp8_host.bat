@echo off
rem Host test driver only: the kernel (fp8_gemm_kernel.cu) is compiled at runtime by NVRTC, because the
rem installed MSVC is newer than CUDA 12.9 supports and cudafe++ crashes on it.
rem Run from the repo root: tests\build_fp8_host.bat
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set CUDA=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9
cl /nologo /O2 /EHsc /std:c++17 /DCUDA_INCLUDE_DIR="\"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/include\"" /I"%CUDA%\include" /I"%~dp0.." /Fo"%~dp0fp8_gemm_host.obj" /Fe"%~dp0fp8_gemm.exe" "%~dp0fp8_gemm_host.cpp" /link /LIBPATH:"%CUDA%\lib\x64" cudart.lib cuda.lib nvrtc.lib
