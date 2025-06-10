@echo off
chcp 65001 >nul
setlocal

echo 神经风格迁移应用打包工具
echo ==============================
echo.

echo 正在检查Python环境...
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo 错误: 未找到Python环境
    pause
    exit /b 1
)
python --version

echo.
echo 正在安装必要的依赖...
python -m pip install pyinstaller
if %errorlevel% neq 0 (
    echo 警告: PyInstaller安装可能失败，继续尝试打包...
)

echo.
echo 开始打包应用...
python build_app.py

echo.
echo 打包完成！按任意键退出...
pause >nul
endlocal