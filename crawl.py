import requests
import random
import time
import re
import csv
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from datetime import datetime
from typing import List, Dict, Optional, Union, Iterator, Literal, Set, Tuple
import logging
from loguru import logger
from prettytable import PrettyTable
from retrying import retry
from urllib.parse import urljoin, urlparse, parse_qs
from collections import deque
import hashlib
import os
from bs4 import BeautifulSoup
import tldextract
import concurrent.futures

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('VoidCrawler')

# 颜色配置
COLORS = {
    'black': '30',
    'red': '31',
    'green': '32',
    'yellow': '33',
    'blue': '34',
    'magenta': '35',
    'cyan': '36',
    'white': '37',
    'reset': '0',
    'bright_black': '90',
    'bright_red': '91',
    'bright_green': '92',
    'bright_yellow': '93',
    'bright_blue': '94',
    'bright_magenta': '95',
    'bright_cyan': '96',
    'bright_white': '97',
}


class Log:
    log = logger

    def set_log_file(self, filepath: str = 'void_crawler.log', rotation: int = 10, level: str = 'DEBUG', color: bool = False):
        self.log.add(
            filepath,
            rotation=f'{rotation} MB',
            colorize=color,
            level=level
        )

    def set_level(self, level):
        self.log.remove()
        logger.add(sys.stdout, level=level)

    @property
    def debug(self):
        return self.__class__.log.debug

    @property
    def info(self):
        return self.__class__.log.info

    @property
    def error(self):
        return self.__class__.log.error

    @property
    def exception(self):
        return self.__class__.log.exception


log = Log()


class RequestObj:
    def __init__(self, url: str, method: Literal['GET', 'POST', 'get', 'post'], headers: Union[None, dict] = None,
                 params: Union[None, dict] = None, data: Union[None, dict] = None, cookies: Union[None, dict] = None,
                 proxys: Union[None, dict] = None, depth: int = 0):
        self.url = url
        self.method = method
        self.headers = headers
        self.params = params
        self.data = data
        self.cookies = cookies
        self.proxys = proxys
        self.depth = depth
        self.request_args = {
            'headers': self.headers,
            'params': self.params,
            'data': self.data,
            'cookies': self.cookies,
            'proxys': self.proxys
        }

    def __repr__(self):
        return f"""
                -------------- request for ----------------
                url  = {self.url}
                method = {self.method}
                depth = {self.depth}
                args = {self.request_args}"""


class ErrorResponse(Exception):
    def __init__(self, message):
        super().__init__(message)

    def __str__(self):
        return f"{self.__class__.__name__}: {self.args[0]}"


class CrawlBase:
    method_processor = {
        'get': requests.get,
        'post': requests.post,
    }
    session = requests.Session()
    session_method_processor = {
        'get': session.get,
        'post': session.post
    }
    file_lock = Lock()

    def __init__(self, sleep_interval: float = .5):
        self.control_sleep_time = sleep_interval

    @retry(
        stop_max_attempt_number=5,
        wait_exponential_multiplier=5000,
        wait_exponential_max=30000,
        retry_on_exception=lambda e: isinstance(e, (requests.RequestException, ErrorResponse))
    )
    def __ajax_requests(self, request_obj: RequestObj, session: bool = False) -> requests.Response:
        if session:
            method_pro = self.session_method_processor[request_obj.method.lower()]
        else:
            method_pro = self.method_processor[request_obj.method.lower()]

        log.debug(f'开始发送请求 {request_obj}')

        if request_obj.method.lower() == 'post':
            resp = method_pro(
                url=request_obj.url,
                headers=request_obj.headers,
                params=request_obj.params,
                data=json.dumps(request_obj.data, ensure_ascii=False, separators=(',', ':')).encode(),
                cookies=request_obj.cookies,
                proxies=request_obj.proxys
            )
        else:
            resp = method_pro(
                url=request_obj.url,
                headers=request_obj.headers,
                params=request_obj.params,
                cookies=request_obj.cookies,
                proxies=request_obj.proxys
            )

        if not self.after_request(request_obj, resp):
            log.error(f'收到的响应数据不符合期望，抛出错误，重试...{request_obj}')
            raise ErrorResponse('用户主动抛弃当前请求')
        return resp

    def before_request(self, request_obj: RequestObj) -> None:
        pass

    def after_request(self, request_obj: RequestObj, response: requests.Response) -> bool:
        return True

    def do_request(self, url: str, method: Literal['GET', 'POST', 'get', 'post'], headers=None, params=None, data=None,
                   cookies=None, proxys=None, session: bool = False, middleware=None, is_abandon: bool = False, depth: int = 0) -> \
    Union[requests.Response, RequestObj, None]:
        if proxys is None:
            proxys = {}
        if cookies is None:
            cookies = {}
        if data is None:
            data = {}
        if params is None:
            params = {}
        if headers is None:
            headers = {}

        request_obj = RequestObj(url=url, method=method, headers=headers, params=params, cookies=cookies, proxys=proxys,
                                 data=data, depth=depth)

        self.before_request(request_obj)

        if isinstance(middleware, list):
            for middle in middleware:
                middle(request_obj)
        elif middleware:
            middleware(request_obj)

        if is_abandon:
            log.debug(f'主动抛弃请求，返回请求对象...{request_obj}')
            return request_obj

        try:
            time.sleep(self.control_sleep_time)
            resp = self.__ajax_requests(request_obj=request_obj, session=session)
            return resp
        except Exception as e:
            log.exception(e)
            return None

    def save_item_to_csv(self, items: Iterator[dict], file_path: str = 'data.csv', max_workers: int = 30) -> None:
        def write_items(item: dict, is_write_headers: bool):
            header = list(item.keys())
            mode = 'a' if not is_write_headers else 'w'
            with self.file_lock:
                with open(file_path, mode, encoding='utf-8', newline='') as fp:
                    writer = csv.DictWriter(fp, header)
                    if is_write_headers:
                        writer.writeheader()
                        log.info(f'写入表头了: {header}, 文件路径: {file_path}')
                    writer.writerow(item)

        is_first = True
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for num, save_item in enumerate(items, start=1):
                executor.submit(write_items, save_item, is_first)
                if is_first:
                    is_first = False
            log.info(f'数据写入完成, 共计{num}条数据！')
            executor.shutdown()

    @staticmethod
    def search_dict(items: dict, search_key: str) -> Iterator:
        stack = [items]
        while stack:
            current_item = stack.pop()
            if isinstance(current_item, dict):
                for key, value in current_item:
                    if search_key == key:
                        yield value
                    else:
                        stack.append(value)
            elif isinstance(current_item, list):
                for value in current_item:
                    stack.append(value)

    @staticmethod
    def format_table_print(items: list[dict], title: str = None, column_names: list = None, *, title_color: str = None,
                           header_color: str = None, column_color: str = None, random_title_color: bool = False,
                           random_header_color: bool = False, random_column_color: bool = False) -> None:
        try:
            assert items and isinstance(items, list) and isinstance(items[0], dict), ValueError(
                "items必须是列表包含字典")
        except AssertionError as e:
            log.exception(e)
            raise ValueError('0')

        def dye_text(text: str, color: str = None, is_random: bool = False) -> str:
            keys = list(COLORS.keys())
            if is_random:
                color = random.choice(keys)
            if color in COLORS:
                return f'\033[1;{COLORS[color]}m{text}\033[0m'
            return text

        max_key = [len(list(col.keys())) for col in items]
        max_key_size = max(max_key)
        item_keys = list(items[max_key.index(max_key_size)].keys())
        column_list = [[] for _ in range(max_key_size)]

        for index, key in enumerate(item_keys):
            for item in items:
                column_list[index].append(item.get(key, ''))

        if not column_names:
            column_names = item_keys

        try:
            assert len(column_names) == len(item_keys), ValueError('表头长度与数据最长项不相符')
        except AssertionError as e:
            log.exception(e)
            raise ValueError('0')

        table = PrettyTable()
        if title:
            table.title = dye_text(title, title_color, random_title_color)

        table.add_column(dye_text('序号', header_color),
                         [dye_text(str(i), column_color, random_column_color) for i in
                          range(1, len(column_list[0]) + 1)])

        for j, column in enumerate(column_list, start=0):
            col = [dye_text(i, column_color, random_column_color) for i in column]
            table.add_column(dye_text(column_names[j], header_color, random_header_color), col)
        print(table)


class FreeProxyPool:
    def __init__(self, test_urls: List[str] = None, check_interval: int = 300, min_proxies: int = 10,
                 max_proxies: int = 50):
        self.test_urls = test_urls or ['https://httpbin.org/ip', 'https://api.ipify.org']
        self.check_interval = check_interval
        self.min_proxies = min_proxies
        self.max_proxies = max_proxies
        self.proxies = []
        self.running = False
        self.lock = threading.Lock()

        # 增强代理源
        self.free_proxy_sources = [
            'https://raw.githubusercontent.com/clarketm/proxy-list/master/proxy-list-raw.txt',
            'https://raw.githubusercontent.com/TheSpeedX/PROXY-List/master/http.txt',
            'https://raw.githubusercontent.com/ShiftyTR/Proxy-List/master/proxy.txt',
            'https://raw.githubusercontent.com/hookzof/socks5_list/master/proxy.txt',
            'https://www.proxy-list.download/api/v1/get?type=http',
            'https://www.proxy-list.download/api/v1/get?type=https',
            'https://www.proxy-list.download/api/v1/get?type=socks4',
            'https://www.proxy-list.download/api/v1/get?type=socks5',
            'https://api.proxyscrape.com/v2/?request=getproxies&protocol=http&timeout=10000&country=all&ssl=all&anonymity=all',
            'https://api.proxyscrape.com/v2/?request=getproxies&protocol=socks4&timeout=10000&country=all&ssl=all&anonymity=all',
            'https://api.proxyscrape.com/v2/?request=getproxies&protocol=socks5&timeout=10000&country=all&ssl=all&anonymity=all'
        ]

        self._start_background_threads()

    def _start_background_threads(self):
        self.running = True
        threading.Thread(target=self._maintain_proxies_loop, daemon=True).start()
        logger.info("Free proxy pool started")

    def _fetch_free_proxies(self) -> List[Dict]:
        all_proxies = []
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

        for source_url in self.free_proxy_sources:
            try:
                response = requests.get(source_url, headers=headers, timeout=10)
                if response.status_code == 200:
                    proxies_from_source = self._parse_proxy_response(response.text, source_url)
                    all_proxies.extend(proxies_from_source)
                    logger.info(f"Fetched {len(proxies_from_source)} proxies from {source_url}")
            except Exception as e:
                logger.warning(f"Failed to fetch from {source_url}: {e}")

        return all_proxies

    def _parse_proxy_response(self, text: str, source_url: str) -> List[Dict]:
        proxies = []
        lines = text.strip().split('\n')

        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            if ':' in line:
                parts = line.split(':')
                if len(parts) >= 2:
                    ip = parts[0].strip()
                    port = parts[1].strip()

                    protocol = 'http'
                    if 'socks5' in source_url.lower() or 'socks5' in line.lower():
                        protocol = 'socks5'
                    elif 'socks4' in source_url.lower() or 'socks4' in line.lower():
                        protocol = 'socks4'
                    elif 'https' in source_url.lower():
                        protocol = 'https'

                    proxies.append({
                        'ip': ip,
                        'port': int(port) if port.isdigit() else 8080,
                        'protocol': protocol,
                        'source': 'free'
                    })

        return proxies

    def _validate_proxy(self, proxy_info: Dict) -> Optional[Dict]:
        proxy_url = f"{proxy_info['protocol']}://{proxy_info['ip']}:{proxy_info['port']}"
        proxies = {proxy_info['protocol']: proxy_url}
        test_url = random.choice(self.test_urls)

        try:
            start_time = time.time()
            response = requests.get(test_url, proxies=proxies, timeout=10,
                                    headers={
                                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})
            response_time = time.time() - start_time

            if response.status_code == 200:
                anonymity = self._check_anonymity(response)
                return {
                    **proxy_info,
                    'response_time': response_time,
                    'last_checked': datetime.now(),
                    'is_valid': True,
                    'anonymity_level': anonymity,
                    'score': self._calculate_score(response_time, anonymity)
                }
        except Exception:
            pass

        return None

    def _check_anonymity(self, response: requests.Response) -> str:
        headers = response.headers
        if 'X-Forwarded-For' in headers or 'Via' in headers:
            return 'transparent'
        elif 'Proxy-Connection' in headers:
            return 'anonymous'
        else:
            return 'elite'

    def _calculate_score(self, response_time: float, anonymity: str) -> int:
        score = 100

        if response_time < 1:
            score += 20
        elif response_time < 3:
            score += 10
        elif response_time > 10:
            score -= 20

        if anonymity == 'elite':
            score += 15
        elif anonymity == 'anonymous':
            score += 5
        else:
            score -= 10

        return max(0, min(100, score))

    def _maintain_proxies_loop(self):
        while self.running:
            try:
                with self.lock:
                    current_count = len([p for p in self.proxies if p.get('is_valid', True)])

                if current_count < self.min_proxies:
                    logger.info(f"Current proxies: {current_count}, fetching new ones...")
                    new_proxies = self._fetch_free_proxies()
                    logger.info(f"Fetched {len(new_proxies)} new proxies")

                    if new_proxies:
                        valid_proxies = self._validate_proxies(new_proxies)

                        with self.lock:
                            self.proxies.extend(valid_proxies)
                            self.proxies = [p for p in self.proxies if
                                            p.get('is_valid', True) and p.get('score', 0) > 30][:self.max_proxies]
                            self.proxies.sort(key=lambda x: x.get('score', 0), reverse=True)

                        logger.info(f"Added {len(valid_proxies)} valid proxies, total: {len(self.proxies)}")

                self._revalidate_existing_proxies()
                time.sleep(self.check_interval)

            except Exception as e:
                logger.error(f"Error in proxy maintenance loop: {e}")
                time.sleep(60)

    def _validate_proxies(self, proxies: List[Dict]) -> List[Dict]:
        valid_proxies = []

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = {executor.submit(self._validate_proxy, proxy): proxy for proxy in proxies}

            for future in futures:
                try:
                    result = future.result(timeout=15)
                    if result:
                        valid_proxies.append(result)
                except:
                    pass

        return valid_proxies

    def _revalidate_existing_proxies(self):
        with self.lock:
            proxies_to_check = self.proxies.copy()

        if not proxies_to_check:
            return

        logger.info(f"Revalidating {len(proxies_to_check)} existing proxies")
        validated_proxies = self._validate_proxies(proxies_to_check)

        with self.lock:
            self.proxies = validated_proxies
            self.proxies.sort(key=lambda x: x.get('score', 0), reverse=True)

        logger.info(f"Revalidation completed, {len(validated_proxies)} proxies remain valid")

    def get_proxy(self, protocol: str = None, min_score: int = 50) -> Optional[Dict]:
        with self.lock:
            candidates = [p for p in self.proxies if p.get('is_valid', True) and p.get('score', 0) >= min_score]
            if protocol:
                candidates = [p for p in candidates if p.get('protocol') == protocol]

        if not candidates:
            logger.warning("No valid proxies available")
            return None

        total_score = sum(p.get('score', 0) for p in candidates)
        if total_score == 0:
            proxy = random.choice(candidates)
        else:
            r = random.uniform(0, total_score)
            current = 0
            for proxy in candidates:
                current += proxy.get('score', 0)
                if r <= current:
                    break

        logger.info(f"Selected proxy: {proxy['ip']}:{proxy['port']} (score: {proxy.get('score', 0)})")
        return proxy

    def report_proxy_status(self, proxy: Dict, success: bool):
        with self.lock:
            for p in self.proxies:
                if (p['ip'] == proxy['ip'] and p['port'] == proxy['port'] and p['protocol'] == proxy['protocol']):
                    if success:
                        p['score'] = min(100, p.get('score', 0) + 5)
                    else:
                        p['score'] = max(0, p.get('score', 0) - 15)
                        p['is_valid'] = p['score'] > 20
                    p['last_used'] = datetime.now()
                    break

    def get_stats(self) -> Dict:
        with self.lock:
            valid_proxies = [p for p in self.proxies if p.get('is_valid', True)]
            all_proxies = self.proxies.copy()

        stats = {
            'total_proxies': len(all_proxies),
            'valid_proxies': len(valid_proxies),
            'by_protocol': {},
            'by_anonymity': {},
            'avg_score': 0,
            'avg_response_time': 0
        }

        if valid_proxies:
            stats['avg_score'] = sum(p.get('score', 0) for p in valid_proxies) / len(valid_proxies)
            response_times = [p.get('response_time', 0) for p in valid_proxies if p.get('response_time')]
            if response_times:
                stats['avg_response_time'] = sum(response_times) / len(response_times)

        for proxy in all_proxies:
            protocol = proxy.get('protocol', 'unknown')
            if protocol not in stats['by_protocol']:
                stats['by_protocol'][protocol] = 0
            stats['by_protocol'][protocol] += 1

        for proxy in all_proxies:
            anonymity = proxy.get('anonymity_level', 'unknown')
            if anonymity not in stats['by_anonymity']:
                stats['by_anonymity'][anonymity] = 0
            stats['by_anonymity'][anonymity] += 1

        return stats

    def stop(self):
        self.running = False
        logger.info("Free proxy pool stopped")


def validate_proxy(proxy_info: Dict, test_urls: List[str] = None) -> Dict:
    test_urls = test_urls or ['https://httpbin.org/ip']
    proxy_url = f"{proxy_info['protocol']}://{proxy_info['ip']}:{proxy_info['port']}"
    proxies = {proxy_info['protocol']: proxy_url}

    result = {**proxy_info, 'is_valid': False, 'response_time': None, 'error': None}

    for test_url in test_urls:
        try:
            start_time = time.time()
            response = requests.get(test_url, proxies=proxies, timeout=10,
                                    headers={
                                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})
            response_time = time.time() - start_time

            if response.status_code == 200:
                result.update({'is_valid': True, 'response_time': response_time, 'last_checked': datetime.now()})
                break
        except Exception as e:
            result['error'] = str(e)

    return result


def batch_validate_proxies(proxies: List[Dict], max_workers: int = 10) -> List[Dict]:
    valid_proxies = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(validate_proxy, proxy) for proxy in proxies]

        for future in futures:
            try:
                result = future.result(timeout=15)
                if result['is_valid']:
                    valid_proxies.append(result)
            except:
                pass

    return valid_proxies


class VoidCrawler(CrawlBase):
    def __init__(self, max_workers: int = 20, request_delay: float = 0.5, use_proxies: bool = True,
                 proxy_list: list = None, max_depth: int = 5, respect_robots_txt: bool = True,
                 user_agent: str = None, crawl_delay: float = 1.0):
        super().__init__(sleep_interval=request_delay)
        self.max_workers = max_workers
        self.request_delay = request_delay
        self.use_proxies = use_proxies
        self.proxy_list = proxy_list or []
        self.current_proxy_index = 0
        self.request_count = 0
        self.success_count = 0
        self.error_count = 0
        self.start_time = None
        self.stats_lock = Lock()
        self.max_depth = max_depth
        self.respect_robots_txt = respect_robots_txt
        self.crawl_delay = crawl_delay

        # URL管理
        self.visited_urls = set()
        self.pending_urls = deque()
        self.url_lock = Lock()
        
        # 域名统计和限制
        self.domain_stats = {}
        self.domain_limits = {}
        self.max_urls_per_domain = 100
        
        # 内容类型优先级
        self.content_priority = {
            'html': 10,
            'text': 9,
            'json': 8,
            'xml': 7,
            'pdf': 5,
            'doc': 4,
            'image': 2,
            'video': 1,
            'audio': 1,
            'other': 0
        }
        
        # URL过滤规则
        self.url_filters = [
            self._filter_by_extension,
            self._filter_by_length,
            self._filter_by_domain_limit,
            self._filter_by_content_type
        ]
        
        # 增强的用户代理列表
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (iPhone; CPU iPhone OS 17_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Mobile/15E148 Safari/604.1',
            'Mozilla/5.0 (iPad; CPU OS 17_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Mobile/15E148 Safari/604.1'
        ]
        
        if user_agent:
            self.user_agents.insert(0, user_agent)
            
        # 初始化代理池
        if use_proxies and not proxy_list:
            self.proxy_pool = FreeProxyPool()
        else:
            self.proxy_pool = None
            
        # 初始化URL优先级队列
        self.priority_urls = []
        self.priority_lock = Lock()
        
        # 创建数据存储目录
        self.data_dir = "void_crawler_data"
        os.makedirs(self.data_dir, exist_ok=True)
        
        # 爬虫状态
        self.is_running = False
        self.pause_event = threading.Event()
        self.pause_event.set()  # 初始为运行状态
        
        # 增强功能：内容提取器
        self.content_extractors = {
            'email': self._extract_emails,
            'phone': self._extract_phone_numbers,
            'id_card': self._extract_id_cards,
            'url': self._extract_urls,
            'social_media': self._extract_social_media
        }
        
        # 增强功能：爬取规则
        self.crawl_rules = {
            'max_retries': 3,
            'timeout': 30,
            'follow_redirects': True,
            'verify_ssl': False,
            'ignore_robots_txt': not respect_robots_txt
        }

    def get_random_headers(self) -> dict:
        return {
            'User-Agent': random.choice(self.user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0'
        }

    def get_next_proxy(self) -> dict:
        if self.proxy_pool:
            proxy = self.proxy_pool.get_proxy()
            if proxy:
                proxy_url = f"{proxy['protocol']}://{proxy['ip']}:{proxy['port']}"
                return {proxy['protocol']: proxy_url}
        
        if not self.proxy_list:
            return {}
            
        proxy = self.proxy_list[self.current_proxy_index]
        self.current_proxy_index = (self.current_proxy_index + 1) % len(self.proxy_list)
        return {'http': proxy, 'https': proxy}

    def update_stats(self, success: bool = True):
        with self.stats_lock:
            self.request_count += 1
            if success:
                self.success_count += 1
            else:
                self.error_count += 1

    def before_request(self, request_obj: RequestObj) -> None:
        if not request_obj.headers:
            request_obj.headers = self.get_random_headers()
        else:
            request_obj.headers.update(self.get_random_headers())

        if self.use_proxies:
            request_obj.proxys = self.get_next_proxy()

    def after_request(self, request_obj: RequestObj, response: requests.Response) -> bool:
        success = response.status_code == 200
        self.update_stats(success)
        if not success:
            log.warning(f"请求失败: {request_obj.url} - 状态码: {response.status_code}")
        return success

    def _normalize_url(self, url: str, base_url: str = None) -> str:
        """标准化URL，处理相对路径和查询参数"""
        if base_url and not url.startswith(('http://', 'https://')):
            url = urljoin(base_url, url)
        
        # 解析URL
        parsed = urlparse(url)
        
        # 移除默认端口
        if (parsed.port == 80 and parsed.scheme == 'http') or (parsed.port == 443 and parsed.scheme == 'https'):
            netloc = parsed.hostname
        else:
            netloc = parsed.netloc
        
        # 移除片段标识符
        url = f"{parsed.scheme}://{netloc}{parsed.path}"
        
        # 处理查询参数 - 按字母顺序排序
        if parsed.query:
            params = parse_qs(parsed.query, keep_blank_values=True)
            sorted_params = sorted(params.items())
            query_string = "&".join([f"{k}={v[0]}" for k, v in sorted_params if v])
            url += f"?{query_string}"
            
        return url

    def _get_url_hash(self, url: str) -> str:
        """获取URL的哈希值，用于去重"""
        return hashlib.md5(url.encode('utf-8')).hexdigest()

    def _get_domain(self, url: str) -> str:
        """获取URL的域名"""
        extracted = tldextract.extract(url)
        return f"{extracted.domain}.{extracted.suffix}"

    def _filter_by_extension(self, url: str) -> bool:
        """根据文件扩展名过滤URL"""
        skip_extensions = [
            '.css', '.js', '.ico', '.png', '.jpg', '.jpeg', '.gif', 
            '.svg', '.woff', '.ttf', '.eot', '.mp3', '.mp4', '.avi',
            '.mov', '.zip', '.rar', '.tar', '.gz', '.7z', '.exe', '.dmg'
        ]
        
        parsed = urlparse(url)
        path = parsed.path.lower()
        
        for ext in skip_extensions:
            if path.endswith(ext):
                return False
        return True

    def _filter_by_length(self, url: str) -> bool:
        """根据URL长度过滤"""
        return len(url) <= 2048

    def _filter_by_domain_limit(self, url: str) -> bool:
        """根据域名限制过滤URL"""
        domain = self._get_domain(url)
        
        with self.url_lock:
            if domain not in self.domain_stats:
                self.domain_stats[domain] = 0
                
            if self.domain_stats[domain] >= self.max_urls_per_domain:
                return False
                
        return True

    def _filter_by_content_type(self, url: str) -> bool:
        """根据内容类型过滤URL"""
        parsed = urlparse(url)
        path = parsed.path.lower()
        
        # 优先爬取HTML页面
        if path.endswith(('.html', '.htm', '.aspx', '.php', '.jsp', '.do')) or '?' in parsed.query:
            return True
            
        # 允许其他文本类型
        if path.endswith(('.txt', '.json', '.xml', '.csv', '.rss', '.atom')):
            return True
            
        # 默认情况下，不爬取媒体文件
        if path.endswith(('.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx')):
            return True
            
        return False

    def _apply_url_filters(self, url: str) -> bool:
        """应用所有URL过滤器"""
        for filter_func in self.url_filters:
            if not filter_func(url):
                return False
        return True

    def _extract_links(self, html: str, base_url: str) -> List[str]:
        """从HTML中提取链接"""
        soup = BeautifulSoup(html, 'html.parser')
        links = set()
        
        # 提取a标签的href属性
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href'].strip()
            if href and not href.startswith(('javascript:', 'mailto:', 'tel:', '#')):
                normalized_url = self._normalize_url(href, base_url)
                if self._apply_url_filters(normalized_url):
                    links.add(normalized_url)
        
        # 提取frame和iframe的src属性
        for frame in soup.find_all(['frame', 'iframe'], src=True):
            src = frame['src'].strip()
            if src:
                normalized_url = self._normalize_url(src, base_url)
                if self._apply_url_filters(normalized_url):
                    links.add(normalized_url)
        
        # 提取link标签的href属性（如RSS、ATOM等）
        for link in soup.find_all('link', href=True):
            href = link['href'].strip()
            if href:
                normalized_url = self._normalize_url(href, base_url)
                if self._apply_url_filters(normalized_url):
                    links.add(normalized_url)
        
        # 提取script标签的src属性
        for script in soup.find_all('script', src=True):
            src = script['src'].strip()
            if src:
                normalized_url = self._normalize_url(src, base_url)
                if self._apply_url_filters(normalized_url):
                    links.add(normalized_url)
        
        return list(links)

    def _calculate_url_priority(self, url: str, depth: int, content_type: str = None) -> int:
        """计算URL优先级"""
        priority = 100  # 基础优先级
        
        # 深度越浅，优先级越高
        priority -= depth * 10
        
        # 根据内容类型调整优先级
        if content_type:
            for ct, score in self.content_priority.items():
                if ct in content_type.lower():
                    priority += score
                    break
        else:
            # 根据URL路径推测内容类型
            parsed = urlparse(url)
            path = parsed.path.lower()
            
            if path.endswith(('.html', '.htm', '.aspx', '.php', '.jsp', '.do')):
                priority += self.content_priority['html']
            elif path.endswith(('.txt', '.json', '.xml', '.csv', '.rss', '.atom')):
                priority += self.content_priority['text']
            elif path.endswith(('.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx')):
                priority += self.content_priority['doc']
            elif any(ext in path for ext in ['.png', '.jpg', '.jpeg', '.gif', '.svg']):
                priority += self.content_priority['image']
            elif any(ext in path for ext in ['.mp3', '.wav', '.ogg']):
                priority += self.content_priority['audio']
            elif any(ext in path for ext in ['.mp4', '.avi', '.mov', '.flv']):
                priority += self.content_priority['video']
        
        # 根据域名统计调整优先级
        domain = self._get_domain(url)
        with self.url_lock:
            if domain in self.domain_stats:
                # 同一域名的URL越多，优先级越低
                priority -= min(50, self.domain_stats[domain] // 10)
        
        return max(0, min(100, priority))

    def _add_url_to_queue(self, url: str, depth: int = 0, priority: int = None):
        """添加URL到爬取队列"""
        url_hash = self._get_url_hash(url)
        
        with self.url_lock:
            if url_hash in self.visited_urls:
                return
                
            self.visited_urls.add(url_hash)
            
            # 更新域名统计
            domain = self._get_domain(url)
            if domain not in self.domain_stats:
                self.domain_stats[domain] = 0
            self.domain_stats[domain] += 1
            
            # 计算优先级
            if priority is None:
                priority = self._calculate_url_priority(url, depth)
            
            # 添加到优先级队列
            with self.priority_lock:
                self.priority_urls.append((priority, depth, url))
                # 保持队列按优先级排序
                self.priority_urls.sort(reverse=True)
            
            log.debug(f"添加URL到队列: {url} (深度: {depth}, 优先级: {priority})")

    def _get_next_url(self) -> Tuple[str, int]:
        """从队列中获取下一个URL"""
        with self.priority_lock:
            if not self.priority_urls:
                return None, -1
                
            # 获取优先级最高的URL
            priority, depth, url = self.priority_urls.pop(0)
            return url, depth

    def _extract_emails(self, text: str) -> List[str]:
        """提取电子邮件地址"""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        return list(set(emails))

    def _extract_phone_numbers(self, text: str) -> List[str]:
        """提取手机号码"""
        phone_patterns = [
            r'1[3-9]\d{9}',
            r'(\+86)?1[3-9]\d{9}',
            r'1[3-9]\d{1,4}[-\s]?\d{1,4}[-\s]?\d{4}',
            r'0\d{2,3}[-\s]?\d{7,8}',
        ]

        phones = []
        for pattern in phone_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                phone = re.sub(r'[^\d]', '', str(match))
                if len(phone) == 11 and phone.startswith(('13', '14', '15', '16', '17', '18', '19')):
                    phones.append(phone)
                elif len(phone) in [10, 11, 12] and phone.startswith('0'):
                    phones.append(phone)

        return list(set(phones))

    def _extract_id_cards(self, text: str) -> List[str]:
        """提取身份证号码"""
        id_pattern = r'[1-9]\d{5}(18|19|20)\d{2}((0[1-9])|(1[0-2]))(([0-2][1-9])|10|20|30|31)\d{3}[0-9Xx]'
        id_cards = []
        matches = re.findall(id_pattern, text)

        for match in matches:
            if isinstance(match, tuple):
                id_card = ''.join(match)
            else:
                id_card = match

            if self.validate_id_card(id_card):
                id_cards.append(id_card.upper())

        return list(set(id_cards))

    def _extract_urls(self, text: str) -> List[str]:
        """从文本中提取URL"""
        url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[/\w\.-]*\??[/\w\.-=&%]*'
        urls = re.findall(url_pattern, text)
        return list(set(urls))

    def _extract_social_media(self, text: str) -> Dict[str, List[str]]:
        """提取社交媒体账号"""
        social_media = {
            'wechat': [],
            'weibo': [],
            'qq': [],
            'douyin': [],
            'kuaishou': []
        }

        # 微信
        wechat_pattern = r'微信号[:：\s]+([a-zA-Z0-9_-]+)'
        wechat_matches = re.findall(wechat_pattern, text)
        social_media['wechat'].extend(wechat_matches)

        # 微博
        weibo_pattern = r'微博[:：\s]+@?([a-zA-Z0-9_-]+)'
        weibo_matches = re.findall(weibo_pattern, text)
        social_media['weibo'].extend(weibo_matches)

        # QQ
        qq_pattern = r'QQ[:：\s]+([0-9]+)'
        qq_matches = re.findall(qq_pattern, text)
        social_media['qq'].extend(qq_matches)

        # 抖音
        douyin_pattern = r'抖音[:：\s]+@?([a-zA-Z0-9_-]+)'
        douyin_matches = re.findall(douyin_pattern, text)
        social_media['douyin'].extend(douyin_matches)

        # 快手
        kuaishou_pattern = r'快手[:：\s]+@?([a-zA-Z0-9_-]+)'
        kuaishou_matches = re.findall(kuaishou_pattern, text)
        social_media['kuaishou'].extend(kuaishou_matches)

        # 去重
        for platform in social_media:
            social_media[platform] = list(set(social_media[platform]))

        return social_media

    def _process_page(self, url: str, depth: int) -> Dict:
        """处理单个页面"""
        result = {
            'url': url,
            'depth': depth,
            'status': 'unknown',
            'content_type': None,
            'content_length': 0,
            'links': [],
            'data': None,
            'error': None,
            'extracted_content': {
                'emails': [],
                'phones': [],
                'id_cards': [],
                'urls': [],
                'social_media': {}
            }
        }
        
        try:
            # 发送请求
            response = self.do_request(url=url, method='GET', depth=depth)
            
            if not response:
                result['status'] = 'error'
                result['error'] = 'No response'
                return result
                
            result['status'] = 'success'
            result['content_type'] = response.headers.get('Content-Type', '').split(';')[0]
            result['content_length'] = len(response.content)
            
            # 如果是HTML页面，提取链接
            if 'text/html' in result['content_type']:
                try:
                    links = self._extract_links(response.text, url)
                    result['links'] = links
                    
                    # 将新链接添加到队列
                    for link in links:
                        if depth + 1 <= self.max_depth:
                            self._add_url_to_queue(link, depth + 1)
                except Exception as e:
                    log.warning(f"提取链接失败: {url} - {e}")
            
            # 保存内容
            try:
                content_type = result['content_type']
                file_ext = '.html'
                
                if 'text/plain' in content_type:
                    file_ext = '.txt'
                elif 'application/json' in content_type:
                    file_ext = '.json'
                elif 'application/xml' in content_type:
                    file_ext = '.xml'
                elif 'application/pdf' in content_type:
                    file_ext = '.pdf'
                
                # 创建基于URL的文件名
                url_hash = self._get_url_hash(url)
                filename = f"{url_hash}{file_ext}"
                filepath = os.path.join(self.data_dir, filename)
                
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                
                result['saved_path'] = filepath
                log.debug(f"保存内容: {filepath}")
            except Exception as e:
                log.warning(f"保存内容失败: {url} - {e}")
                
            # 提取页面数据
            if 'text/html' in content_type or 'text/plain' in content_type:
                text_content = response.text
                result['data'] = text_content
                
                # 提取各种内容
                result['extracted_content']['emails'] = self._extract_emails(text_content)
                result['extracted_content']['phones'] = self._extract_phone_numbers(text_content)
                result['extracted_content']['id_cards'] = self._extract_id_cards(text_content)
                result['extracted_content']['urls'] = self._extract_urls(text_content)
                result['extracted_content']['social_media'] = self._extract_social_media(text_content)
                
                # 保存提取的内容
                self._save_extracted_content(url, result['extracted_content'])
            else:
                result['data'] = f"<Binary data: {result['content_length']} bytes>"
                
        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)
            log.error(f"处理页面失败: {url} - {e}")
            
        return result

    def _save_extracted_content(self, url: str, extracted_content: Dict):
        """保存提取的内容"""
        url_hash = self._get_url_hash(url)
        
        # 保存电子邮件
        if extracted_content['emails']:
            email_file = os.path.join(self.data_dir, f"{url_hash}_emails.csv")
            with open(email_file, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Email', 'Source URL', 'Extract Time'])
                for email in extracted_content['emails']:
                    writer.writerow([email, url, datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
        
        # 保存手机号
        if extracted_content['phones']:
            phone_file = os.path.join(self.data_dir, f"{url_hash}_phones.csv")
            with open(phone_file, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Phone', 'Source URL', 'Extract Time'])
                for phone in extracted_content['phones']:
                    writer.writerow([phone, url, datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
        
        # 保存身份证信息
        if extracted_content['id_cards']:
            id_file = os.path.join(self.data_dir, f"{url_hash}_id_cards.csv")
            with open(id_file, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['ID Card', 'Source URL', 'Extract Time'])
                for id_card in extracted_content['id_cards']:
                    writer.writerow([id_card, url, datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
        
        # 保存社交媒体账号
        if any(extracted_content['social_media'].values()):
            social_file = os.path.join(self.data_dir, f"{url_hash}_social_media.csv")
            with open(social_file, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Platform', 'Account', 'Source URL', 'Extract Time'])
                for platform, accounts in extracted_content['social_media'].items():
                    for account in accounts:
                        writer.writerow([platform, account, url, datetime.now().strftime('%Y-%m-%d %H:%M:%S')])

    def start_crawling(self, seed_urls: List[str], max_pages: int = 1000):
        """开始爬取"""
        self.is_running = True
        self.start_time = time.time()
        
        # 添加种子URL到队列
        for url in seed_urls:
            self._add_url_to_queue(url, 0, 100)  # 种子URL优先级最高
        
        log.info(f"开始虚空爬取，种子URL: {len(seed_urls)}，最大页面数: {max_pages}")
        
        # 创建线程池
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            processed_count = 0
            
            while self.is_running and processed_count < max_pages:
                # 检查暂停状态
                self.pause_event.wait()
                
                # 获取下一个URL
                url, depth = self._get_next_url()
                if not url:
                    if not futures:  # 没有正在处理的任务且队列为空
                        break
                    time.sleep(0.1)
                    continue
                
                # 提交任务
                future = executor.submit(self._process_page, url, depth)
                futures.append(future)
                
                # 限制并发任务数量
                if len(futures) >= self.max_workers * 2:
                    completed, _ = concurrent.futures.wait(
                        futures, return_when=concurrent.futures.FIRST_COMPLETED)
                    for f in completed:
                        try:
                            result = f.result()
                            processed_count += 1
                            
                            # 定期打印统计信息
                            if processed_count % 50 == 0:
                                self._print_stats()
                        except Exception as e:
                            log.error(f"任务执行失败: {e}")
                    
                    # 从列表中移除已完成的任务
                    futures = [f for f in futures if not f.done()]
                
                # 控制添加新URL的速度
                time.sleep(self.request_delay)
            
            # 等待所有任务完成
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    processed_count += 1
                except Exception as e:
                    log.error(f"任务执行失败: {e}")
        
        self._print_stats()
        log.info(f"虚空爬取完成，共处理 {processed_count} 个页面")

    def pause_crawling(self):
        """暂停爬取"""
        self.pause_event.clear()
        log.info("爬取已暂停")

    def resume_crawling(self):
        """恢复爬取"""
        self.pause_event.set()
        log.info("爬取已恢复")

    def stop_crawling(self):
        """停止爬取"""
        self.is_running = False
        self.pause_event.set()
        log.info("爬取已停止")

    def _print_stats(self):
        """打印统计信息"""
        if self.start_time:
            elapsed_time = time.time() - self.start_time
            requests_per_second = self.request_count / elapsed_time if elapsed_time > 0 else 0

            with self.url_lock:
                total_urls = len(self.visited_urls)
                pending_urls = len(self.priority_urls)
                top_domains = sorted(self.domain_stats.items(), key=lambda x: x[1], reverse=True)[:5]

            log.info(f"""
=== 虚空爬虫统计信息 ===
总请求数: {self.request_count}
成功请求: {self.success_count}
失败请求: {self.error_count}
成功率: {self.success_count / self.request_count * 100:.2f}%
已访问URL: {total_urls}
待处理URL: {pending_urls}
爬取深度: {self.max_depth}
总耗时: {elapsed_time:.2f}秒
平均请求速度: {requests_per_second:.2f} 请求/秒
热门域名: {', '.join([f'{domain}({count})' for domain, count in top_domains])}
==================
            """)

    def set_max_depth(self, depth: int):
        """设置最大爬取深度"""
        self.max_depth = depth
        log.info(f"最大爬取深度已设置为: {depth}")

    def set_max_urls_per_domain(self, limit: int):
        """设置每个域名的最大URL数量"""
        self.max_urls_per_domain = limit
        log.info(f"每个域名的最大URL数量已设置为: {limit}")

    def set_domain_limit(self, domain: str, limit: int):
        """设置特定域名的URL限制"""
        self.domain_limits[domain] = limit
        log.info(f"域名 {domain} 的URL限制已设置为: {limit}")

    def validate_id_card(self, id_card: str) -> bool:
        """验证身份证号码"""
        if len(id_card) != 18:
            return False

        factors = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]
        check_codes = ['1', '0', 'X', '9', '8', '7', '6', '5', '4', '3', '2']

        try:
            total = sum(int(id_card[i]) * factors[i] for i in range(17))
            check_code = check_codes[total % 11]
            return id_card[17].upper() == check_code
        except:
            return False

    def parse_id_card_info(self, id_card: str) -> Dict[str, str]:
        """解析身份证信息"""
        if not self.validate_id_card(id_card):
            return {}

        area_code = id_card[:6]
        birth_date = id_card[6:14]

        try:
            birth_datetime = datetime.strptime(birth_date, '%Y%m%d')
            birth_formatted = birth_datetime.strftime('%Y-%m-%d')
            age = datetime.now().year - birth_datetime.year
        except:
            birth_formatted = birth_date
            age = '未知'

        gender_num = int(id_card[16])
        gender = '男' if gender_num % 2 == 1 else '女'

        return {
            'id_card': id_card,
            'area_code': area_code,
            'birth_date': birth_formatted,
            'age': str(age),
            'gender': gender,
            'is_valid': '是'
        }

    def crawl_phone_and_id_from_urls(self, urls: List[str], method: str = 'GET', save_to_file: str = None) -> Dict[
        str, List]:
        """从URL中爬取手机号和身份证信息"""
        results = self.crawl_urls_concurrently(urls, method)

        all_phones = []
        all_id_cards = []
        detailed_id_info = []

        for result in results:
            if result['status'] == 'success' and result['data']:
                text = result['data'] if isinstance(result['data'], str) else str(result['data'])

                phones = self._extract_phone_numbers(text)
                all_phones.extend(phones)

                id_cards = self._extract_id_cards(text)
                all_id_cards.extend(id_cards)

                for id_card in id_cards:
                    id_info = self.parse_id_card_info(id_card)
                    if id_info:
                        id_info.update({
                            'source_url': result['url'],
                            'crawl_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        })
                        detailed_id_info.append(id_info)

        all_phones = list(set(all_phones))
        all_id_cards = list(set(all_id_cards))

        if save_to_file:
            self._save_phone_id_data(all_phones, detailed_id_info, save_to_file)

        return {
            'phones': all_phones,
            'id_cards': all_id_cards,
            'detailed_id_info': detailed_id_info
        }

    def _save_phone_id_data(self, phones: List[str], id_info: List[Dict], filename: str):
        """保存手机号和身份证数据"""
        if phones:
            phone_file = f'phones_{filename}'
            with open(phone_file, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['手机号', '提取时间'])
                for phone in phones:
                    writer.writerow([phone, datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
            log.info(f'手机号数据已保存到: {phone_file}')

        if id_info:
            id_file = f'id_cards_{filename}'
            with open(id_file, 'w', encoding='utf-8', newline='') as f:
                if id_info:
                    fieldnames = list(id_info[0].keys())
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(id_info)
            log.info(f'身份证数据已保存到: {id_file}')

    def crawl_from_file(self, file_path: str, url_pattern: str = r'https?://[^\s]+', save_to_file: str = None) -> Dict[
        str, List]:
        """从文件中读取URL并进行爬取"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            urls = re.findall(url_pattern, content)
            urls = list(set(urls))

            if not urls:
                log.warning('未在文件中找到URL')
                return {'phones': [], 'id_cards': [], 'detailed_id_info': []}

            log.info(f'从文件中提取到 {len(urls)} 个URL')
            return self.crawl_phone_and_id_from_urls(urls, save_to_file=save_to_file)

        except Exception as e:
            log.error(f'读取文件失败: {e}')
            return {'phones': [], 'id_cards': [], 'detailed_id_info': []}

    def crawl_urls_concurrently(self, urls: list, method: str = 'GET', params_list: list = None, data_list: list = None,
                                callback: callable = None) -> list:
        """并发爬取URL列表"""
        if params_list is None:
            params_list = [{}] * len(urls)
        if data_list is None:
            data_list = [{}] * len(urls)

        self.start_time = time.time()
        results = []
        result_lock = Lock()

        def process_url(url_index: int):
            url = urls[url_index]
            params = params_list[url_index]
            data = data_list[url_index]

            try:
                response = self.do_request(url=url, method=method, params=params, data=data)
                if response and response.status_code == 200:
                    result = {
                        'url': url,
                        'status': 'success',
                        'response': response,
                        'data': response.text if callback is None else callback(response)
                    }
                else:
                    result = {
                        'url': url,
                        'status': 'error',
                        'response': response,
                        'data': None
                    }

                with result_lock:
                    results.append(result)

                time.sleep(random.uniform(self.request_delay, self.request_delay * 2))

            except Exception as e:
                log.error(f"处理URL时发生错误: {url} - {e}")
                with result_lock:
                    results.append({
                        'url': url,
                        'status': 'exception',
                        'response': None,
                        'data': None,
                        'error': str(e)
                    })

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(process_url, i) for i in range(len(urls))]
            for future in futures:
                future.result()

        self._print_stats()
        return results

    def crawl_with_pagination(self, base_url: str, page_param: str, start_page: int, end_page: int, method: str = 'GET',
                              callback: callable = None) -> list:
        """爬取分页内容"""
        urls = []
        params_list = []

        for page in range(start_page, end_page + 1):
            urls.append(base_url)
            params_list.append({page_param: str(page)})

        return self.crawl_urls_concurrently(urls, method, params_list, callback=callback)

    def set_proxy_list(self, proxy_list: list):
        """设置代理列表"""
        self.proxy_list = proxy_list
        self.use_proxies = bool(proxy_list)

    def set_max_workers(self, max_workers: int):
        """设置最大工作线程数"""
        self.max_workers = max_workers

    def set_request_delay(self, delay: float):
        """设置请求延迟"""
        self.request_delay = delay
        self.control_sleep_time = delay

    def generate_crawl_report(self) -> Dict:
        """生成爬取报告"""
        if not self.start_time:
            return {"error": "爬取尚未开始"}

        elapsed_time = time.time() - self.start_time
        requests_per_second = self.request_count / elapsed_time if elapsed_time > 0 else 0

        with self.url_lock:
            total_urls = len(self.visited_urls)
            pending_urls = len(self.priority_urls)
            top_domains = sorted(self.domain_stats.items(), key=lambda x: x[1], reverse=True)[:10]

        report = {
            "summary": {
                "total_requests": self.request_count,
                "successful_requests": self.success_count,
                "failed_requests": self.error_count,
                "success_rate": f"{self.success_count / self.request_count * 100:.2f}%" if self.request_count > 0 else "0%",
                "visited_urls": total_urls,
                "pending_urls": pending_urls,
                "max_depth": self.max_depth,
                "elapsed_time": f"{elapsed_time:.2f}秒",
                "requests_per_second": f"{requests_per_second:.2f} 请求/秒"
            },
            "top_domains": [{"domain": domain, "count": count} for domain, count in top_domains],
            "proxy_stats": self.proxy_pool.get_stats() if self.proxy_pool else {},
            "data_dir": self.data_dir
        }

        return report


def demo_void_crawler():
    """虚空爬虫演示"""
    # 创建虚空爬虫实例
    crawler = VoidCrawler(
        max_workers=10,
        request_delay=0.5,
        use_proxies=True,
        max_depth=3,
        respect_robots_txt=True
    )
    
    # 设置种子URL
    seed_urls = [
        'https://www.baidu.com',
        'https://www.qq.com',
        'https://www.sina.com.cn'
    ]
    
    # 开始爬取
    crawler.start_crawling(seed_urls, max_pages=100)
    
    # 打印最终统计信息
    crawler._print_stats()
    
    # 生成爬取报告
    report = crawler.generate_crawl_report()
    print("\n爬取报告:")
    print(json.dumps(report, indent=2, ensure_ascii=False))


def main():
    """主函数"""
    print("虚空爬虫 - 爬取整个网络世界的角落技术")
    print("=" * 50)
    
    # 创建虚空爬虫实例
    crawler = VoidCrawler(
        max_workers=15,
        request_delay=0.3,
        use_proxies=True,
        max_depth=4,
        respect_robots_txt=True
    )
    
    # 设置种子URL
    seed_urls = [
        'https://www.baidu.com',
        'https://www.qq.com',
        'https://www.sina.com.cn',
        'https://www.163.com',
        'https://www.sohu.com'
    ]
    
    # 开始爬取
    crawler.start_crawling(seed_urls, max_pages=500)
    
    # 打印最终统计信息
    crawler._print_stats()
    
    # 生成爬取报告
    report = crawler.generate_crawl_report()
    print("\n爬取报告:")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
