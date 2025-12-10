"""
后端启动脚本 - 自动清理端口并启动服务
"""
import subprocess
import sys
import os
import socket
import time

PORT = 8102


def is_port_available(port: int) -> bool:
    """检查端口是否可用"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('0.0.0.0', port))
        sock.close()
        return True
    except OSError:
        return False


def kill_port_process(port: int):
    """杀掉占用指定端口的进程"""
    result = subprocess.run(
        f'netstat -ano | findstr ":{port}" | findstr "LISTENING"',
        shell=True, capture_output=True, text=True
    )

    pids = set()
    for line in result.stdout.strip().split('\n'):
        if line.strip():
            parts = line.split()
            if len(parts) >= 5:
                pid = parts[-1]
                if pid.isdigit() and pid != '0':
                    pids.add(pid)

    if pids:
        print(f"发现 {len(pids)} 个进程占用端口 {port}，正在清理...")
        for pid in pids:
            try:
                subprocess.run(f'taskkill /F /PID {pid}', shell=True, capture_output=True)
                print(f"  已终止进程 PID={pid}")
            except:
                pass
        return True
    return False


def wait_for_port(port: int, timeout: int = 10) -> bool:
    """等待端口可用"""
    print(f"等待端口 {port} 释放...", end='', flush=True)
    start = time.time()
    while time.time() - start < timeout:
        if is_port_available(port):
            print(" 就绪!")
            return True
        print(".", end='', flush=True)
        time.sleep(0.5)
    print(" 超时!")
    return False


def main():
    print(f"\n{'='*50}")
    print(f"Tier6+ 拓扑可视化后端")
    print(f"{'='*50}\n")

    # 切换到脚本所在目录
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    port = PORT

    # 如果端口被占用，先尝试清理
    if not is_port_available(port):
        kill_port_process(port)

        # 等待端口释放
        if not wait_for_port(port, timeout=5):
            # 如果还是不行，使用备用端口
            for alt_port in range(PORT + 1, PORT + 100):
                if is_port_available(alt_port):
                    port = alt_port
                    print(f"使用备用端口: {port}")
                    break
            else:
                print("错误: 无法找到可用端口")
                sys.exit(1)

    # 启动服务
    print(f"\n启动服务在端口 {port}...")
    print(f"API地址: http://localhost:{port}")

    if port != PORT:
        print(f"\n注意: 使用了备用端口 {port}，请修改前端配置或等待端口 {PORT} 释放后重试")

    print(f"按 Ctrl+C 停止服务\n")

    # 启动uvicorn
    import uvicorn
    from main import app
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
