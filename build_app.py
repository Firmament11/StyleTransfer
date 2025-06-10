import os
import sys
import subprocess
import shutil
from pathlib import Path

def build_executable():
    """
    使用PyInstaller将Flask应用打包成可执行文件
    """
    print("开始构建可执行文件...")
    
    # 确保PyInstaller已安装
    try:
        import PyInstaller
    except ImportError:
        print("正在安装PyInstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
    
    # 创建spec文件内容
    spec_content = """
# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('templates', 'templates'),
        ('static', 'static'),
        ('styles', 'styles'),
        ('history', 'history'),
        ('uploads', 'uploads'),
    ],
    hiddenimports=[
        'torch',
        'torchvision',
        'PIL',
        'flask',
        'numpy',
        'cv2'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='neural_style_transfer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='neural_style_transfer',
)
"""
    
    # 写入spec文件
    with open('neural_style_transfer.spec', 'w', encoding='utf-8') as f:
        f.write(spec_content)
    
    # 运行PyInstaller
    try:
        cmd = [sys.executable, '-m', 'PyInstaller', 'neural_style_transfer.spec', '--clean']
        subprocess.run(cmd, check=True)
        print("\n构建成功！")
        print("可执行文件位置: dist/neural_style_transfer/")
        print("\n使用说明:")
        print("1. 将整个 dist/neural_style_transfer/ 文件夹复制到目标电脑")
        print("2. 运行 neural_style_transfer.exe")
        print("3. 在浏览器中访问 http://localhost:5000")
        
    except subprocess.CalledProcessError as e:
        print(f"构建失败: {e}")
        return False
    
    return True

def create_installer():
    """
    创建安装程序脚本
    """
    installer_script = """
@echo off
echo 神经风格迁移应用安装程序
echo ==============================

set INSTALL_DIR=%USERPROFILE%\\NeuralStyleTransfer

echo 正在创建安装目录...
if not exist "%INSTALL_DIR%" mkdir "%INSTALL_DIR%"

echo 正在复制文件...
xcopy /E /I /Y "neural_style_transfer" "%INSTALL_DIR%"

echo 创建桌面快捷方式...
set SHORTCUT_PATH=%USERPROFILE%\\Desktop\\神经风格迁移.lnk
set TARGET_PATH=%INSTALL_DIR%\\neural_style_transfer.exe

powershell "$WshShell = New-Object -comObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut('%SHORTCUT_PATH%'); $Shortcut.TargetPath = '%TARGET_PATH%'; $Shortcut.WorkingDirectory = '%INSTALL_DIR%'; $Shortcut.Save()"

echo 安装完成！
echo 您可以通过桌面快捷方式启动应用
echo 或者直接运行: %INSTALL_DIR%\\neural_style_transfer.exe
pause
"""
    
    with open('dist/install.bat', 'w', encoding='gbk') as f:
        f.write(installer_script)
    
    print("安装脚本已创建: dist/install.bat")

def create_readme():
    """
    创建使用说明文件
    """
    readme_content = """
# 神经风格迁移应用

## 安装说明

### 方法一：使用安装脚本（推荐）
1. 解压所有文件到任意文件夹
2. 双击运行 `install.bat`
3. 按照提示完成安装
4. 使用桌面快捷方式启动应用

### 方法二：手动安装
1. 将 `neural_style_transfer` 文件夹复制到您希望的位置
2. 双击 `neural_style_transfer.exe` 启动应用
3. 在浏览器中访问 http://localhost:5000

## 使用说明

1. **上传图片**：
   - 内容图片：您想要应用风格的原始图片
   - 风格图片：提供艺术风格的参考图片
   - 支持拖拽上传和点击上传

2. **调整参数**：
   - 样式权重：控制风格迁移的强度（0.1-1.0）
   - 迭代步数：处理质量，步数越多质量越好但耗时更长

3. **开始处理**：
   - 点击"开始风格迁移"按钮
   - 等待处理完成
   - 查看结果并下载

4. **历史记录**：
   - 查看之前的处理结果
   - 下载历史图片
   - 删除不需要的记录

## 系统要求

- Windows 10 或更高版本
- 至少 4GB 内存
- 建议使用独立显卡以获得更好性能

## 故障排除

1. **应用无法启动**：
   - 确保所有文件完整
   - 以管理员身份运行
   - 检查防火墙设置

2. **处理速度慢**：
   - 降低迭代步数
   - 使用较小的图片
   - 确保电脑有足够内存

3. **浏览器无法访问**：
   - 确认应用已启动
   - 尝试访问 http://127.0.0.1:5000
   - 检查端口5000是否被占用

## 技术支持

如有问题，请检查应用目录下的日志文件。
"""
    
    with open('dist/README.md', 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print("使用说明已创建: dist/README.md")

if __name__ == "__main__":
    print("神经风格迁移应用打包工具")
    print("=" * 30)
    
    # 检查必要文件
    if not os.path.exists('main.py'):
        print("错误: 找不到 main.py 文件")
        sys.exit(1)
    
    # 构建可执行文件
    if build_executable():
        # 创建安装脚本和说明文件
        create_installer()
        create_readme()
        
        print("\n" + "="*50)
        print("打包完成！")
        print("\n发布包内容:")
        print("- dist/neural_style_transfer/ (应用文件夹)")
        print("- dist/install.bat (安装脚本)")
        print("- dist/README.md (使用说明)")
        print("\n您可以将整个 dist 文件夹打包分发给其他用户。")
    else:
        print("打包失败，请检查错误信息。")