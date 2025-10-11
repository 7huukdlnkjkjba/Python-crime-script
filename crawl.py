import requests
import random
import time
import re
import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from datetime import datetime
from typing import List, Dict, Optional, Union, Iterator, Literal
import logging
from loguru import logger
from prettytable import PrettyTable
from retrying import retry
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import os

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('FreeProxyPool')

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

    def set_log_file(self, filepath: str = 'log.log', rotation: int = 10, level: str = 'DEBUG', color: bool = False):
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
                 proxys: Union[None, dict] = None):
        self.url = url
        self.method = method
        self.headers = headers
        self.params = params
        self.data = data
        self.cookies = cookies
        self.proxys = proxys
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
                   cookies=None, proxys=None, session: bool = False, middleware=None, is_abandon: bool = False) -> \
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
                                 data=data)

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

        self.free_proxy_sources = [
            'https://raw.githubusercontent.com/clarketm/proxy-list/master/proxy-list-raw.txt',
            'https://raw.githubusercontent.com/TheSpeedX/PROXY-List/master/http.txt',
            'https://raw.githubusercontent.com/ShiftyTR/Proxy-List/master/proxy.txt',
            'https://raw.githubusercontent.com/hookzof/socks5_list/master/proxy.txt',
            'https://www.proxy-list.download/api/v1/get?type=http',
            'https://www.proxy-list.download/api/v1/get?type=https',
            'https://www.proxy-list.download/api/v1/get?type=socks4',
            'https://www.proxy-list.download/api/v1/get?type=socks5'
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


class HighConcurrencyCrawler(CrawlBase):
    def __init__(self, max_workers: int = 10, request_delay: float = 0.1, use_proxies: bool = False,
                 proxy_list: list = None):
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

        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 Edg/91.0.864.59'
        ]

    def get_random_headers(self) -> dict:
        return {
            'User-Agent': random.choice(self.user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }

    def get_next_proxy(self) -> dict:
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

        if self.use_proxies and self.proxy_list:
            request_obj.proxys = self.get_next_proxy()

    def after_request(self, request_obj: RequestObj, response: requests.Response) -> bool:
        success = response.status_code == 200
        self.update_stats(success)
        if not success:
            log.warning(f"请求失败: {request_obj.url} - 状态码: {response.status_code}")
        return success

    def crawl_urls_concurrently(self, urls: list, method: str = 'GET', params_list: list = None, data_list: list = None,
                                callback: callable = None) -> list:
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
        urls = []
        params_list = []

        for page in range(start_page, end_page + 1):
            urls.append(base_url)
            params_list.append({page_param: str(page)})

        return self.crawl_urls_concurrently(urls, method, params_list, callback=callback)

    def _print_stats(self):
        if self.start_time:
            elapsed_time = time.time() - self.start_time
            requests_per_second = self.request_count / elapsed_time if elapsed_time > 0 else 0

            log.info(f"""
=== 爬虫统计信息 ===
总请求数: {self.request_count}
成功请求: {self.success_count}
失败请求: {self.error_count}
成功率: {self.success_count / self.request_count * 100:.2f}%
总耗时: {elapsed_time:.2f}秒
平均请求速度: {requests_per_second:.2f} 请求/秒
==================
            """)

    def set_proxy_list(self, proxy_list: list):
        self.proxy_list = proxy_list
        self.use_proxies = bool(proxy_list)

    def set_max_workers(self, max_workers: int):
        self.max_workers = max_workers

    def set_request_delay(self, delay: float):
        self.request_delay = delay
        self.control_sleep_time = delay

    def extract_phone_numbers(self, text: str) -> List[str]:
        phone_patterns = [
            r'1[3-9]\d{9}',
            r'(\+86)?1[3-9]\d{9}',
            r'1[3-9]\d{1,4}[-\s]?\d{1,4}[-\s]?\d{4}',
        ]

        phones = []
        for pattern in phone_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                phone = re.sub(r'[^\d]', '', str(match))
                if len(phone) == 11 and phone.startswith(('13', '14', '15', '16', '17', '18', '19')):
                    phones.append(phone)

        return list(set(phones))

    def extract_id_cards(self, text: str) -> List[str]:
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

    def validate_id_card(self, id_card: str) -> bool:
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
        results = self.crawl_urls_concurrently(urls, method)

        all_phones = []
        all_id_cards = []
        detailed_id_info = []

        for result in results:
            if result['status'] == 'success' and result['data']:
                text = result['data'] if isinstance(result['data'], str) else str(result['data'])

                phones = self.extract_phone_numbers(text)
                all_phones.extend(phones)

                id_cards = self.extract_id_cards(text)
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


def demo_high_concurrency_crawler():
    crawler = HighConcurrencyCrawler(max_workers=10, request_delay=0.0001, use_proxies=False)

    urls = ['https://www.baidu.com', 'https://www.qq.com', 'https://www.taobao.com'] * 3

    results = crawler.crawl_urls_concurrently(urls)

    for result in results:
        print(f"URL: {result['url']}, 状态: {result['status']}")

    test_text = """
    我的手机号是13812345678，另一个是+8613912345678，
    身份证号是110101199001011234，另一个是11010119900307987X
    """

    phones = crawler.extract_phone_numbers(test_text)
    print(f"提取到的手机号: {phones}")

    id_cards = crawler.extract_id_cards(test_text)
    print(f"提取到的身份证: {id_cards}")

    for id_card in id_cards:
        info = crawler.parse_id_card_info(id_card)
        print(f"身份证信息: {info}")


class GeetestCaptchaCracker:
    """验证码破解类，借鉴crack_geetest-master项目的验证码破解逻辑"""

    def __init__(self, mainimg_checkpoint_path=None, verimg_checkpoint_path=None, output_dir='./output/'):
        """
        初始化验证码破解器

        Args:
            mainimg_checkpoint_path: 主图片模型路径
            verimg_checkpoint_path: 验证码图片模型路径
            output_dir: 输出目录
        """
        self.mainimg_checkpoint_path = mainimg_checkpoint_path or './output/chinese/mainimg/'
        self.verimg_checkpoint_path = verimg_checkpoint_path or './output/chinese/verimg/'
        self.output_dir = output_dir

        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'main_img'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'verificate_img'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'chinese'), exist_ok=True)

        # 初始化模型
        self._initialize_models()

    def _load_label_dict(self):
        """加载标签字典"""
        label_dict_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       'GeetChinese_crack-master', 'data', 'label_dict.txt')

        if not os.path.exists(label_dict_path):
            # 如果标签字典不存在，创建一个默认的
            label_dict = {
                "0": "一", "1": "二", "2": "三", "3": "四", "4": "五",
                "5": "六", "6": "七", "7": "八", "8": "九", "9": "十",
                "10": "百", "11": "千", "12": "万", "13": "亿", "14": "零"
            }
            # 保存标签字典
            os.makedirs(os.path.dirname(label_dict_path), exist_ok=True)
            with open(label_dict_path, 'w', encoding='utf-8') as f:
                for k, v in label_dict.items():
                    f.write(f"{k}:{v}\n")
        else:
            # 加载标签字典
            label_dict = {}
            with open(label_dict_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if ':' in line:
                        k, v = line.strip().split(':', 1)
                        label_dict[k] = v

        return label_dict

    def _initialize_models(self):
        """初始化模型"""
        try:
            # 导入边界框检测模块
            sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'GeetChinese_crack-master'))
            import Chinese_bboxdetect as bboxdetect
            import Chinese_OCR_new as orc

            # 加载Faster R-CNN网络
            self.bboxdetect_sess, self.bboxdetect_net = bboxdetect.load_faster_rcnn_network()

            # 加载OCR网络
            self.mainimg_graph = tf.Graph()
            self.mainimg_sess, self.mainimg_net = orc.load_chinese_orc_net(
                self.mainimg_graph, self.mainimg_checkpoint_path)

            self.verimg_graph = tf.Graph()
            self.verimg_sess, self.verimg_net = orc.load_chinese_orc_net(
                self.verimg_graph, self.verimg_checkpoint_path)

            # 加载标签字典
            self.label_dict = self._load_label_dict()

            log.info("验证码破解模型初始化成功")
        except Exception as e:
            log.error(f"验证码破解模型初始化失败: {e}")
            raise

    def _find_image_bbox(self, img):
        """查找图像边界框"""
        v_sum = np.sum(img, axis=0)
        start_i = None
        end_i = None
        minimun_range = 10
        maximun_range = 20
        min_val = 10
        peek_ranges = []
        ser_val = 0

        # 从左往右扫描，遇到非零像素点就以此为字体的左边界
        for i, val in enumerate(v_sum):
            # 定位第一个字体的起始位置
            if val > min_val and start_i is None:
                start_i = i
                ser_val = 0
            # 继续扫描到字体，继续往右扫描
            elif val > min_val and start_i is not None:
                ser_val = 0
            # 扫描到背景，判断空白长度
            elif val <= min_val and start_i is not None:
                ser_val = ser_val + 1
                if (i - start_i >= minimun_range and ser_val > 2) or (i - start_i >= maximun_range):
                    end_i = i
                    if start_i > 5:
                        start_i = start_i - 5
                    peek_ranges.append((start_i, end_i + 2))
                    start_i = None
                    end_i = None
            # 扫描到背景，继续扫描下一个字体
            elif val <= min_val and start_i is None:
                ser_val = ser_val + 1
            else:
                raise ValueError("cannot parse this case...")

        return peek_ranges

    def _cut_geetcode(self, img_path):
        """切割验证码图片"""
        path = img_path.replace('\\', '/')
        image = Image.open(path)
        main_box = (0, 0, 334, 343)
        ver_box = (0, 344, 115, 384)

        log.info(f"处理图片: {path}")

        # 切割主图片
        region = image.crop(main_box)
        main_img_path = os.path.join(self.output_dir, 'main_img', 'main_' + path.split('/')[-1])
        region.save(main_img_path)

        # 切割验证码图片
        region = image.crop(ver_box)
        ver_img_path = os.path.join(self.output_dir, 'verificate_img', 'verificate_' + path.split('/')[-1])
        region.save(ver_img_path)

        return main_img_path, ver_img_path

    def _run_vercode_boxdetect(self, ver_img):
        """运行验证码边界框检测"""
        ver_img = ver_img.replace('\\', '/')
        image = cv2.imread(ver_img, cv2.IMREAD_GRAYSCALE)
        ret, image1 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
        box = self._find_image_bbox(image1)

        imagename = ver_img.split('/')[-1].split('.')[0]
        box_index = 0
        vercode_box_info = []
        cls_vercode_box_info = []
        bbox = np.array([0, 0, 0, 0])

        for starx, endx in box:
            box_index = box_index + 1
            region = image[0:40, starx:endx]
            out_path = os.path.join(self.output_dir, 'chinese', imagename + '_' + str(box_index) + '.jpg')
            cv2.imwrite(out_path, region)
            vercode_box_info.append([out_path, bbox])

        cls_vercode_box_info.append(vercode_box_info)
        return cls_vercode_box_info

    def _judge_orc_result(self, geetcode_bbox_predict_top3, vercode_bbox_predict_top3):
        """判断OCR结果"""
        Chinese_Orc_bbox = []

        for vercode_index, ver_bbox in enumerate(vercode_bbox_predict_top3):
            ver_top3_index = ver_bbox[1]
            entry = {
                'vercode_index': -1,
                'predict_index': -1,
                'predict_dict': -1,
                'predict_box': [0, 0, 0, 0],
                'predict_success': False
            }

            entry['vercode_index'] = vercode_index
            entry['predict_success'] = False

            for geetcode_index, geet_bbox in enumerate(geetcode_bbox_predict_top3):
                geet_top3_index = geet_bbox[1]
                for ver_pre_index in ver_top3_index:
                    if ver_pre_index in geet_top3_index:
                        entry['predict_index'] = ver_pre_index
                        entry['predict_dict'] = self.label_dict.get(str(ver_pre_index), '未知')
                        entry['predict_box'] = geet_bbox[0][1]
                        entry['predict_success'] = True
                        break

                if entry['predict_success'] is True:
                    geetcode_bbox_predict_top3.remove(geet_bbox)
                    break

            Chinese_Orc_bbox.append(entry)

        # 为未匹配的验证码分配预测结果
        for Orc_index, Orc_txt in enumerate(Chinese_Orc_bbox):
            if Orc_txt['predict_success'] is False:
                try:
                    geet_bbox = geetcode_bbox_predict_top3[0]
                    Orc_txt['predict_box'] = geet_bbox[0][1]
                    geetcode_bbox_predict_top3.remove(geet_bbox)
                except:
                    log.warning('预测失败')

        return Chinese_Orc_bbox

    def crack_captcha(self, img_path):
        """
        破解单个验证码

        Args:
            img_path: 验证码图片路径

        Returns:
            dict: 破解结果，包含验证码字符和位置信息
        """
        try:
            # 切割验证码图片
            main_img, ver_img = self._cut_geetcode(img_path)

            # 导入边界框检测和OCR模块
            sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'GeetChinese_crack-master'))
            import Chinese_bboxdetect as bboxdetect
            import Chinese_OCR_new as orc

            # 运行边界框检测
            geetcode_bbox = bboxdetect.run_geetcode_boxdetect(main_img,
                                                              os.path.join(self.output_dir, 'chinese'),
                                                              self.bboxdetect_sess, self.bboxdetect_net)

            # 运行验证码边界框检测
            vercode_bbox = self._run_vercode_boxdetect(ver_img)

            # 运行OCR识别
            geetcode_bbox_predict_top3 = orc.run_chinese_orc(self.mainimg_sess, self.mainimg_net, geetcode_bbox)
            vercode_bbox_predict_top3 = orc.run_chinese_orc(self.verimg_sess, self.verimg_net, vercode_bbox)

            # 判断OCR结果
            Chinese_Orc_bbox = self._judge_orc_result(geetcode_bbox_predict_top3, vercode_bbox_predict_top3)

            # 提取验证码字符
            captcha_chars = []
            for item in Chinese_Orc_bbox:
                if item['predict_success']:
                    captcha_chars.append(item['predict_dict'])

            # 构建结果
            result = {
                'success': True,
                'captcha_text': ''.join(captcha_chars),
                'details': Chinese_Orc_bbox,
                'main_img_path': main_img,
                'ver_img_path': ver_img
            }

            log.info(f"验证码破解成功: {result['captcha_text']}")
            return result

        except Exception as e:
            log.error(f"验证码破解失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'captcha_text': '',
                'details': []
            }

    def batch_crack_captchas(self, img_paths):
        """
        批量破解验证码

        Args:
            img_paths: 验证码图片路径列表

        Returns:
            list: 破解结果列表
        """
        results = []

        for img_path in img_paths:
            result = self.crack_captcha(img_path)
            results.append(result)

        return results

    def close(self):
        """关闭会话，释放资源"""
        try:
            if hasattr(self, 'bboxdetect_sess'):
                self.bboxdetect_sess.close()
            if hasattr(self, 'mainimg_sess'):
                self.mainimg_sess.close()
            if hasattr(self, 'verimg_sess'):
                self.verimg_sess.close()
            log.info("验证码破解器资源已释放")
        except Exception as e:
            log.error(f"释放资源时出错: {e}")


def demo_geetest_captcha_cracker():
    """演示验证码破解功能"""
    # 创建验证码破解器实例
    cracker = GeetestCaptchaCracker()

    # 单个验证码破解示例
    # 注意：这里需要替换为实际的验证码图片路径
    sample_captcha_path = './data/demo/input/sample.jpg'

    if os.path.exists(sample_captcha_path):
        result = cracker.crack_captcha(sample_captcha_path)
        print(f"单个验证码破解结果: {result}")
    else:
        print(f"示例验证码图片不存在: {sample_captcha_path}")

    # 批量验证码破解示例
    captcha_dir = './data/demo/input/'
    if os.path.exists(captcha_dir):
        captcha_files = [os.path.join(captcha_dir, f) for f in os.listdir(captcha_dir)
                         if f.endswith(('.jpg', '.png', '.jpeg'))]

        if captcha_files:
            batch_results = cracker.batch_crack_captchas(captcha_files)
            print(f"批量破解了 {len(batch_results)} 个验证码")
            for i, result in enumerate(batch_results):
                print(f"验证码 {i + 1}: {result['captcha_text'] if result['success'] else '破解失败'}")
        else:
            print("未找到验证码图片文件")
    else:
        print(f"验证码目录不存在: {captcha_dir}")

    # 关闭破解器，释放资源
    cracker.close()


def main():
    """主函数"""
    print("选择要运行的功能:")
    print("1. 高并发爬虫演示")
    print("2. 验证码破解演示")

    choice = input("请输入选项(1/2): ").strip()

    if choice == '1':
        demo_high_concurrency_crawler()
    elif choice == '2':
        demo_geetest_captcha_cracker()
    else:
        print("无效选项，运行高并发爬虫演示")
        demo_high_concurrency_crawler()


if __name__ == '__main__':
    main()