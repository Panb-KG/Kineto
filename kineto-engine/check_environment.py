#!/usr/bin/env python3
"""
Kineto Engine 环境自动配置检查脚本
自动检测系统环境并提示相应的配置步骤
"""

import sys
import platform
import subprocess
from pathlib import Path

def print_section(title):
    """打印带分隔线的标题"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")

def check_python_version():
    """检查 Python 版本"""
    print("🔍 检查 Python 版本...")
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    print(f"   当前版本: Python {version_str}")
    
    if version.major == 3 and version.minor >= 10:
        print(f"   ✅ Python 版本符合要求 (≥ 3.10)")
        return True
    else:
        print(f"   ❌ Python 版本过低，需要 3.10+")
        return False

def check_conda():
    """检查 Conda 是否安装"""
    print("🔍 检查 Conda 环境...")
    try:
        result = subprocess.run(['conda', '--version'], capture_output=True, text=True)
        print(f"   ✅ {result.stdout.strip()}")
        return True
    except FileNotFoundError:
        print("   ❌ Conda 未安装，请先安装 Miniconda 或 Anaconda")
        print("   📥 下载地址: https://docs.conda.io/projects/miniconda/en/latest/")
        return False

def check_os_and_gpu():
    """检查操作系统和 GPU 支持"""
    print("🔍 检查系统环境...")
    
    os_name = platform.system()
    print(f"   系统: {os_name}")
    print(f"   处理器: {platform.processor()}")
    
    # 检测 CUDA
    try:
        result = subprocess.run(['nvidia-smi', '--version'], capture_output=True, text=True)
        print(f"   ✅ NVIDIA CUDA 检测到")
        print(f"      推荐: pip install torch --index-url https://download.pytorch.org/whl/cu121")
        return 'cuda'
    except FileNotFoundError:
        pass
    
    # 检测 Apple Silicon
    if os_name == 'Darwin':
        try:
            result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                  capture_output=True, text=True)
            cpu_info = result.stdout.strip()
            if 'Apple' in cpu_info or 'M1' in cpu_info or 'M2' in cpu_info:
                print(f"   ✅ Apple Silicon (MPS) 检测到")
                print(f"      推荐: pip install torch torchvision torchaudio")
                return 'mps'
        except:
            pass
    
    print(f"   ℹ️  未检测到 CUDA/MPS，将使用 CPU 模式（推理缓慢）")
    return 'cpu'

def check_pytorch():
    """检查 PyTorch 是否安装"""
    print("🔍 检查 PyTorch...")
    try:
        import torch
        print(f"   ✅ PyTorch {torch.__version__} 已安装")
        
        if torch.cuda.is_available():
            print(f"   ✅ CUDA 支持已启用")
            print(f"      GPU: {torch.cuda.get_device_name(0)}")
            print(f"      显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print(f"   ✅ MPS 支持已启用 (Apple Silicon)")
        else:
            print(f"   ℹ️  仅支持 CPU 推理")
        
        return True
    except ImportError:
        print("   ❌ PyTorch 未安装")
        return False

def check_opencv():
    """检查 OpenCV"""
    print("🔍 检查 OpenCV...")
    try:
        import cv2
        print(f"   ✅ OpenCV {cv2.__version__} 已安装")
        return True
    except ImportError:
        print("   ❌ OpenCV 未安装")
        return False

def check_dependencies():
    """检查其他关键依赖"""
    print("🔍 检查其他依赖...")
    
    dependencies = {
        'numpy': 'NumPy',
        'pydantic': 'Pydantic',
        'requests': 'Requests',
        'fastapi': 'FastAPI',
        'smpl_x': 'SMPL-X',
    }
    
    all_ok = True
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"   ✅ {name}")
        except ImportError:
            print(f"   ⚠️  {name} 未安装")
            all_ok = False
    
    return all_ok

def main():
    print_section("🚀 Kineto Engine 环境检查工具")
    
    checks = [
        ("Python 版本", check_python_version),
        ("Conda 环境", check_conda),
        ("系统与 GPU", check_os_and_gpu),
        ("PyTorch", check_pytorch),
        ("OpenCV", check_opencv),
        ("依赖库", check_dependencies),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"   ❌ 检查失败: {e}")
            results.append((name, False))
    
    # 总结
    print_section("📋 检查总结")
    
    all_ok = all(result for _, result in results)
    
    for name, result in results:
        status = "✅" if result else "❌"
        print(f"{status} {name}")
    
    print()
    
    if all_ok:
        print("✨ 所有检查通过！环境已就绪。")
        print("\n📝 后续步骤：")
        print("   1. 准备测试视频 input_video.mp4")
        print("   2. 运行: python kineto_core.py --input input_video.mp4 --output pose_data.json")
    else:
        print("⚠️  部分检查未通过。请按以下步骤配置：")
        print("\n📋 推荐配置步骤：")
        print("   1. conda create -n kineto-engine python=3.11 -y")
        print("   2. conda activate kineto-engine")
        
        gpu_type = check_os_and_gpu()
        if gpu_type == 'cuda':
            print("   3. pip install torch --index-url https://download.pytorch.org/whl/cu121")
        else:
            print("   3. pip install torch torchvision torchaudio")
        
        print("   4. pip install -r requirements.txt")
        print("   5. git clone https://github.com/shubham-goel/4D-Humans.git")
        print("   6. cd 4D-Humans && pip install -e .")
        
        print("\n📖 详细指南请查看: SETUP_GUIDE.md")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  用户取消")
        sys.exit(1)
