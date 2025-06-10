import os
import sys

# 设置环境变量解决OpenMP冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

try:
    # 导入main模块并启动应用
    from main import app
    
    print("神经风格迁移应用启动中...")
    print("PWA功能已启用")
    print("请在浏览器中访问: http://localhost:5000")
    print("按 Ctrl+C 停止应用")
    print("-" * 50)
    
    # 启动Flask应用
    app.run(host='0.0.0.0', port=5000, debug=False)
    
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保所有依赖已正确安装")
    input("按回车键退出...")
except Exception as e:
    print(f"启动失败: {e}")
    input("按回车键退出...")