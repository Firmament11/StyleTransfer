# 神经风格迁移 PWA 应用打包指南

## PWA 功能

本项目已经转换为 Progressive Web App (PWA)，具备以下功能：

### 🌟 PWA 特性
- **离线访问**：通过 Service Worker 缓存关键资源
- **应用图标**：支持添加到主屏幕
- **响应式设计**：适配各种设备屏幕
- **原生应用体验**：全屏显示，无浏览器地址栏

### 📱 如何安装 PWA
1. 在支持的浏览器中打开应用
2. 点击地址栏中的"安装"按钮
3. 或者通过浏览器菜单选择"添加到主屏幕"

## 打包为独立应用

### 🚀 快速打包

**方法一：使用批处理脚本（推荐）**
```bash
# 双击运行 build.bat 文件
build.bat
```

**方法二：手动命令行打包**
```bash
# 安装 PyInstaller
pip install pyinstaller

# 使用 spec 文件打包
pyinstaller build.spec
```

### 📋 打包前准备

1. **确保依赖完整**
   ```bash
   pip install -r requirements.txt
   ```

2. **检查图标文件**
   - `static/icon-192x192.png`
   - `static/icon-512x512.png`
   - `static/favicon.svg`

3. **验证文件夹结构**
   ```
   项目根目录/
   ├── main.py
   ├── build.spec
   ├── build.bat
   ├── requirements.txt
   ├── templates/
   │   └── balba.html
   ├── static/
   │   ├── manifest.json
   │   ├── sw.js
   │   ├── favicon.svg
   │   ├── icon-192x192.png
   │   └── icon-512x512.png
   ├── styles/
   ├── uploads/
   └── history/
   ```

### 🔧 自定义打包配置

编辑 `build.spec` 文件可以自定义打包选项：

```python
# 修改应用名称
name='神经风格迁移',

# 添加图标（需要 .ico 文件）
icon='static/favicon.ico',

# 控制台模式（True=显示控制台，False=隐藏控制台）
console=True,
```

### 📦 打包输出

打包成功后，在 `dist/` 文件夹中找到：
- `神经风格迁移.exe` - 主执行文件
- 相关依赖文件和资源

### 🚀 部署到其他电脑

1. **复制整个 dist 文件夹**到目标电脑
2. **运行 `神经风格迁移.exe`**
3. **首次运行**可能需要较长时间解压资源

### ⚠️ 注意事项

1. **GPU 支持**
   - 如果目标电脑没有 CUDA 驱动，应用会自动切换到 CPU 模式
   - GPU 模式需要目标电脑安装对应的 CUDA 和 cuDNN

2. **文件大小**
   - 打包后的应用较大（通常 > 1GB），因为包含了 PyTorch 等深度学习库
   - 建议使用压缩工具减小分发文件大小

3. **性能考虑**
   - CPU 模式运行速度较慢，建议在有 GPU 的电脑上使用
   - 可以通过应用内的设备信息查看当前使用的计算设备

4. **防火墙设置**
   - 应用会在本地启动 Web 服务器，可能需要防火墙允许
   - 默认端口：5000

### 🔍 故障排除

**打包失败**
- 检查是否安装了所有依赖：`pip install -r requirements.txt`
- 确保 Python 版本兼容（推荐 Python 3.8+）
- 检查磁盘空间是否充足

**运行时错误**
- 查看控制台输出的错误信息
- 确保所有资源文件都在正确位置
- 检查是否有杀毒软件误报

**性能问题**
- 首次运行较慢是正常现象
- 如果持续缓慢，检查是否在使用 CPU 模式
- 可以通过应用界面查看设备信息

### 📞 技术支持

如果遇到问题，请检查：
1. 错误日志信息
2. 系统环境（Python 版本、CUDA 版本等）
3. 打包时的完整输出信息

---

**享受您的神经风格迁移应用！** 🎨✨