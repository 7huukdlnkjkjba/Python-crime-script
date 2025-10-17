#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
crawl.py — 全部整合版 · 内核稳定（核心版）
适配 Python 3.12+

设计目标（核心）：
- 每一次抓取都有明确目标（无“虚空”遍历）
- AI 参与“是否深入”的决策（可插拔）
- 结构化输出（JSONL），可接数据库或消息队列
- 高可用内核：Session 重用、TaskGroup、域断路器、优雅关机
- 明确扩展点：AI / Blockchain / Storage

作者：AI助手（重构示例）
时间：2025
"""

from __future__ import annotations

import asyncio
import aiohttp
import logging
import sys
import os
import time
import json
import random
import signal
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List, Iterable, Callable, Tuple
from urllib.parse import urlparse, urljoin
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import urllib.robotparser

# Optional libraries
try:
    from loguru import logger as _logger
    _logger.remove()
    _logger.add(sys.stdout, format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | {message}")
    logger = _logger
except Exception:
    logger = logging.getLogger("corecrawler")
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Optional uvloop
try:
    if sys.platform != "win32":
        import uvloop
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        logger.info("uvloop enabled")
except Exception as e:
    logger.debug(f"uvloop not used: {e}")

# tenacity for robust retry
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# prometheus optional
try:
    from prometheus_client import start_http_server, Counter, Gauge
    PROMETHEUS_AVAILABLE = True
except Exception:
    PROMETHEUS_AVAILABLE = False

# ---------------------------
# Configuration (集中配置)
# ---------------------------
class Config:
    # 基本
    MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "50"))
    DEFAULT_TIMEOUT = int(os.getenv("DEFAULT_TIMEOUT", "20"))
    SESSION_LIMIT_PER_HOST = int(os.getenv("SESSION_LIMIT_PER_HOST", "10"))
    USER_AGENT = os.getenv("USER_AGENT", "CoreCrawler/2025 (+https://example.local)")

    # 重试
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
    RETRY_BASE = float(os.getenv("RETRY_BASE", "1"))

    # 域限速
    DEFAULT_PER_DOMAIN_INTERVAL = float(os.getenv("DEFAULT_PER_DOMAIN_INTERVAL", "0.5"))

    # robots.txt
    ROBOTS_TTL = int(os.getenv("ROBOTS_TTL", "3600"))

    # proxy
    PROXY_VALIDATION_TIMEOUT = int(os.getenv("PROXY_VALIDATION_TIMEOUT", "8"))
    PROXY_MIN_SCORE = int(os.getenv("PROXY_MIN_SCORE", "30"))

    # storage
    OUTPUT_JSONL = os.getenv("OUTPUT_JSONL", "crawl_output.jsonl")

    # circuit breaker
    CB_FAILURE_THRESHOLD = int(os.getenv("CB_FAILURE_THRESHOLD", "5"))
    CB_RECOVER_SECONDS = int(os.getenv("CB_RECOVER_SECONDS", "300"))

    # prometheus
    PROM_PORT = int(os.getenv("PROM_PORT", "8001"))

config = Config()

# ---------------------------
# Metrics（Prometheus 可选）
# ---------------------------
if PROMETHEUS_AVAILABLE:
    MET_REQUESTS = Counter("corecrawler_requests_total", "Total requests attempted")
    MET_SUCCESS = Counter("corecrawler_success_total", "Successful (2xx) requests")
    MET_FAILED = Counter("corecrawler_failed_total", "Failed requests")
    MET_PROXY_POOL = Gauge("corecrawler_proxy_pool_size", "Proxy pool size")
    try:
        start_http_server(config.PROM_PORT)
        logger.info(f"Prometheus metrics listening on :{config.PROM_PORT}")
    except Exception as e:
        logger.warning(f"Prometheus server failed to start: {e}")

# ---------------------------
# Exceptions
# ---------------------------
class CoreCrawlerError(Exception):
    pass

class CircuitOpenError(CoreCrawlerError):
    pass

class ProxyPoolEmpty(CoreCrawlerError):
    pass

# ---------------------------
# Data structures
# ---------------------------
@dataclass
class CrawlTask:
    """任务的最小描述：url + purpose + 元信息"""
    url: str
    purpose: str  # 描述任务意图，例如 "extract_article" / "get_price"
    depth: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CrawlResult:
    url: str
    timestamp: str
    status: int
    elapsed: float
    data: Dict[str, Any]  # 结构化抽取后的数据
    tx_hash: Optional[str] = None  # 区块链存证占位

# ---------------------------
# Session 管理
# ---------------------------
class SessionManager:
    """
    统一管理 aiohttp ClientSession。
    优点：连接复用、统一超时与 connector 配置。
    """
    def __init__(self, timeout: int = config.DEFAULT_TIMEOUT, limit_per_host: int = config.SESSION_LIMIT_PER_HOST):
        self._timeout = aiohttp.ClientTimeout(total=timeout)
        self._connector = aiohttp.TCPConnector(limit_per_host=limit_per_host)
        self._session: Optional[aiohttp.ClientSession] = None
        self._lock = asyncio.Lock()

    async def session(self) -> aiohttp.ClientSession:
        async with self._lock:
            if self._session is None or self._session.closed:
                self._session = aiohttp.ClientSession(timeout=self._timeout, connector=self._connector)
                logger.info("Created aiohttp session")
            return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
            logger.info("Closed aiohttp session")

# ---------------------------
# robots.txt 缓存
# ---------------------------
class RobotsCache:
    """
    robots.txt 缓存解析器（基于 urllib.robotparser）。
    每个域名缓存一个 parser，TTL 可配置。
    """
    def __init__(self, ttl: int = config.ROBOTS_TTL):
        self._ttl = ttl
        self._cache: Dict[str, Tuple[urllib.robotparser.RobotFileParser, float]] = {}
        self._lock = asyncio.Lock()

    async def allowed(self, url: str, ua: str = config.USER_AGENT) -> bool:
        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        async with self._lock:
            entry = self._cache.get(origin)
            if entry and time.time() - entry[1] < self._ttl:
                rp = entry[0]
            else:
                rp = urllib.robotparser.RobotFileParser()
                try:
                    rp.set_url(urljoin(origin, "/robots.txt"))
                    rp.read()
                except Exception as e:
                    logger.debug(f"robots fetch error {origin}: {e}")
                self._cache[origin] = (rp, time.time())
        try:
            return rp.can_fetch(ua, url)
        except Exception:
            # 若解析异常，为了不把爬虫全部阻断，保守选择允许。
            return True

# ---------------------------
# Per-domain adaptive rate limiter & circuit breaker
# ---------------------------
class CircuitBreaker:
    """
    简单域级断路器：
    - failure_count >= threshold -> open
    - open 状态持续 recover_seconds 后允许半开尝试
    """
    def __init__(self, failure_threshold: int = config.CB_FAILURE_THRESHOLD, recover_seconds: int = config.CB_RECOVER_SECONDS):
        self.failure_threshold = failure_threshold
        self.recover_seconds = recover_seconds
        self._state: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def before(self, domain: str):
        async with self._lock:
            state = self._state.get(domain)
            if not state:
                return
            if state.get("open"):
                opened_at = state.get("opened_at", 0)
                if time.time() - opened_at < self.recover_seconds:
                    raise CircuitOpenError(f"circuit open for domain {domain}")
                else:
                    # 半开：允许一次尝试并清除 fail_count
                    state["open"] = False
                    state["fail_count"] = 0
                    logger.info(f"Circuit half-open for {domain}")

    async def success(self, domain: str):
        async with self._lock:
            state = self._state.setdefault(domain, {"fail_count": 0, "open": False, "opened_at": 0})
            state["fail_count"] = 0
            state["open"] = False

    async def failure(self, domain: str):
        async with self._lock:
            state = self._state.setdefault(domain, {"fail_count": 0, "open": False, "opened_at": 0})
            state["fail_count"] += 1
            if state["fail_count"] >= self.failure_threshold:
                state["open"] = True
                state["opened_at"] = time.time()
                logger.warning(f"Circuit opened for domain {domain} due to {state['fail_count']} failures")

class DomainRateLimiter:
    """
    每个域名最小时间间隔限制，支持动态调整（如 AI 调度器更新 interval）。
    """
    def __init__(self, default_interval: float = config.DEFAULT_PER_DOMAIN_INTERVAL):
        self._intervals: Dict[str, float] = {}
        self._last_request: Dict[str, float] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
        self._default = default_interval

    async def wait(self, domain: str):
        lock = self._locks.setdefault(domain, asyncio.Lock())
        async with lock:
            last = self._last_request.get(domain, 0)
            interval = self._intervals.get(domain, self._default)
            now = time.time()
            wait_for = max(0.0, interval - (now - last))
            if wait_for > 0:
                await asyncio.sleep(wait_for)
            self._last_request[domain] = time.time()

    def set_interval(self, domain: str, interval: float):
        self._intervals[domain] = interval
        logger.debug(f"Set domain {domain} interval={interval:.3f}s")

# ---------------------------
# Proxy Pool（轻量）
# ---------------------------
@dataclass
class Proxy:
    ip: str
    port: int
    protocol: str = "http"
    score: int = 50
    last_checked: Optional[float] = None
    is_valid: bool = True

    def url(self) -> str:
        return f"{self.protocol}://{self.ip}:{self.port}"

class ProxyPool:
    """
    轻量代理池，实现：
    - 批量添加（并发验证）
    - get (weighted sampling)
    - report(success/fail) 调整分数
    """
    def __init__(self, session_mgr: SessionManager, validate_urls: Optional[List[str]] = None):
        self._proxies: List[Proxy] = []
        self._lock = asyncio.Lock()
        self._session_mgr = session_mgr
        self._validate_urls = validate_urls or ["https://httpbin.org/ip", "https://api.ipify.org"]
        self._validate_sem = asyncio.Semaphore(20)

    async def add(self, proxy_specs: Iterable[Dict[str, Any]]):
        tasks = []
        for s in proxy_specs:
            p = Proxy(ip=s["ip"], port=int(s.get("port", 80)), protocol=s.get("protocol", "http"))
            tasks.append(self._validate_and_add(p))
        await asyncio.gather(*tasks, return_exceptions=True)
        if PROMETHEUS_AVAILABLE:
            MET_PROXY_POOL.set(len(self._proxies))

    async def _validate_and_add(self, proxy: Proxy):
        ok = await self.validate(proxy)
        if ok:
            async with self._lock:
                # dedupe
                if not any((p.ip, p.port, p.protocol) == (proxy.ip, proxy.port, proxy.protocol) for p in self._proxies):
                    self._proxies.append(proxy)
        return ok

    async def validate(self, proxy: Proxy) -> bool:
        async with self._validate_sem:
            session = await self._session_mgr.session()
            url = random.choice(self._validate_urls)
            try:
                async with session.get(url, proxy=proxy.url(), timeout=aiohttp.ClientTimeout(total=config.PROXY_VALIDATION_TIMEOUT), headers={"User-Agent": config.USER_AGENT}) as resp:
                    proxy.last_checked = time.time()
                    if resp.status == 200:
                        proxy.score = min(100, proxy.score + 20)
                        proxy.is_valid = proxy.score >= config.PROXY_MIN_SCORE
                        return proxy.is_valid
                    else:
                        proxy.score = max(0, proxy.score - 30)
                        proxy.is_valid = False
                        return False
            except Exception:
                proxy.score = max(0, proxy.score - 40)
                proxy.is_valid = False
                proxy.last_checked = time.time()
                return False

    async def get(self, min_score: int = config.PROXY_MIN_SCORE) -> Proxy:
        async with self._lock:
            candidates = [p for p in self._proxies if p.is_valid and p.score >= min_score]
            if not candidates:
                candidates = [p for p in self._proxies if p.is_valid]
            if not candidates:
                raise ProxyPoolEmpty("no valid proxies")
            total = sum(p.score for p in candidates)
            r = random.uniform(0, total)
            upto = 0
            for p in candidates:
                upto += p.score
                if r <= upto:
                    return p
            return candidates[-1]

    async def report(self, proxy: Proxy, success: bool):
        async with self._lock:
            for p in self._proxies:
                if (p.ip, p.port, p.protocol) == (proxy.ip, proxy.port, proxy.protocol):
                    if success:
                        p.score = min(100, p.score + 5)
                    else:
                        p.score = max(0, p.score - 15)
                        if p.score < 10:
                            p.is_valid = False
                    p.last_checked = time.time()
                    break

# ---------------------------
# AI Scheduler（可插拔）
# ---------------------------
class BaseAIScheduler:
    """
    抽象 AI 调度器接口。
    - should_continue(html, url, metadata) -> bool: 是否继续深入（True/False）
    - suggest_interval(domain, stats) -> float: 建议域间隔（秒）
    可扩展为 LLM / 分类器等。
    """
    async def should_continue(self, html: str, url: str, metadata: Dict[str, Any]) -> bool:
        raise NotImplementedError

    async def suggest_interval(self, domain: str, stats: Dict[str, Any]) -> Optional[float]:
        raise NotImplementedError

class HeuristicAIScheduler(BaseAIScheduler):
    """
    轻量启发式实现（可立即使用）：
    - 基于页面长度、包含关键词、禁止词、重复/噪声判断
    """
    def __init__(self, min_len: int = 500, stop_words: Optional[List[str]] = None, important_words: Optional[List[str]] = None):
        self.min_len = min_len
        self.stop_words = stop_words or ["login", "signup", "404", "captcha", "register", "error"]
        self.important_words = important_words or ["article", "新闻", "price", "产品", "contact", "about"]

    async def should_continue(self, html: str, url: str, metadata: Dict[str, Any]) -> bool:
        txt = (BeautifulSoup(html, "lxml").get_text(" ", strip=True) if html else "")
        if not txt or len(txt) < self.min_len:
            return False
        low = txt.lower()
        for w in self.stop_words:
            if w in low:
                return False
        # 若包含重要词则优先深入
        for iw in self.important_words:
            if iw.lower() in low:
                return True
        # 默认深度判断：若足够长则允许
        return len(low) > self.min_len * 1.2

    async def suggest_interval(self, domain: str, stats: Dict[str, Any]) -> Optional[float]:
        # 简单实现：根据失败率扩大间隔
        failures = stats.get("failures", 0)
        successes = stats.get("successes", 1)
        fail_rate = failures / max(1, (failures + successes))
        base = config.DEFAULT_PER_DOMAIN_INTERVAL
        return min(10.0, base * (1 + fail_rate * 5))

class LLM_AIScheduler(BaseAIScheduler):
    """
    LLM 驱动实现（占位）：
    - 将 HTML 摘要或页面元信息发到 LLM，LLM 返回是否继续与推荐间隔。
    - 你可以把此处替换为调用 OpenAI/Transformers 的实现。
    """
    def __init__(self, llm_call: Optional[Callable[[str], Dict[str, Any]]] = None):
        """
        llm_call: 可调用对象，接收 prompt (str) 或结构化数据，返回 dict { 'continue': bool, 'interval': float }
        """
        self.llm_call = llm_call

    async def should_continue(self, html: str, url: str, metadata: Dict[str, Any]) -> bool:
        if not self.llm_call:
            # fallback conservative
            return False
        prompt = f"Decide if the following page {url} is worth deeper crawling. Return JSON with 'continue': true/false.\n\nHtml snippet:\n{html[:2000]}"
        try:
            resp = self.llm_call(prompt)
            return bool(resp.get("continue", False))
        except Exception as e:
            logger.warning(f"LLM call failed in should_continue: {e}")
            return False

    async def suggest_interval(self, domain: str, stats: Dict[str, Any]) -> Optional[float]:
        if not self.llm_call:
            return None
        prompt = f"Given domain {domain} stats {json.dumps(stats)}, suggest a per-domain interval in seconds."
        try:
            resp = self.llm_call(prompt)
            return float(resp.get("interval", config.DEFAULT_PER_DOMAIN_INTERVAL))
        except Exception:
            return None

# ---------------------------
# Structured Extractor（可扩展）
# ---------------------------
class StructuredExtractor:
    """
    抽取规则集合：将 HTML -> 结构化 dict
    默认提供：article (title, text), meta tags
    你可以扩展或注入专门的解析器 (e.g. schema.org, OpenGraph, 商城价格解析)
    """
    def __init__(self):
        pass

    def extract(self, html: str, url: str, purpose: str) -> Dict[str, Any]:
        soup = BeautifulSoup(html, "lxml") if html else BeautifulSoup("", "html.parser")
        result: Dict[str, Any] = {"url": url, "purpose": purpose}
        # 标题
        title = soup.find("title")
        if title:
            result["title"] = title.get_text(strip=True)
        # meta description
        desc = soup.find("meta", attrs={"name": "description"}) or soup.find("meta", attrs={"property": "og:description"})
        if desc and desc.get("content"):
            result["description"] = desc.get("content").strip()
        # article text
        paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        content = " ".join(p for p in paragraphs if p)
        if content:
            result["content"] = content[:20000]  # 截断以避免过大
        # other structured data placeholders
        # e.g. schema.org, price extraction etc.
        return result

# ---------------------------
# Blockchain Integration 占位（可插入真实实现）
# ---------------------------
class BlockchainIntegration:
    async def store_crawl_proof(self, url: str, payload_hash: str) -> str:
        """
        占位实现：返回伪 tx hash
        在真实接入时，用 web3.py 调用智能合约或发送交易，并返回真实 tx hash。
        """
        tx_hash = hashlib.sha256((url + payload_hash + str(time.time())).encode()).hexdigest()
        logger.debug(f"Blockchain placeholder stored for {url} tx={tx_hash[:12]}...")
        return tx_hash

# ---------------------------
# 核心内核：Goal-Driven Crawler (稳定内核)
# ---------------------------
class CoreCrawler:
    """
    稳定内核：
    - 任务队列（CrawlTask）
    - 明确目的（purpose）驱动链接跟进策略
    - AI 决策参与（should_continue / suggest_interval）
    - 域断路器 + per-domain rate limiter
    - 结构化输出到 JSONL（或注入自定义 Storage）
    """
    def __init__(self,
                 session_mgr: SessionManager,
                 robots: RobotsCache,
                 rate_limiter: DomainRateLimiter,
                 cb: CircuitBreaker,
                 extractor: StructuredExtractor,
                 ai_scheduler: Optional[BaseAIScheduler] = None,
                 proxy_pool: Optional[ProxyPool] = None,
                 blockchain: Optional[BlockchainIntegration] = None,
                 output_jsonl: str = config.OUTPUT_JSONL):
        self.session_mgr = session_mgr
        self.robots = robots
        self.rate_limiter = rate_limiter
        self.cb = cb
        self.extractor = extractor
        self.ai = ai_scheduler or HeuristicAIScheduler()
        self.proxy_pool = proxy_pool
        self.blockchain = blockchain
        self.output_file = output_jsonl

        self._semaphore = asyncio.Semaphore(config.MAX_CONCURRENCY)
        self._task_queue: asyncio.Queue[CrawlTask] = asyncio.Queue()
        self._visited: set[str] = set()
        self._stop_requested = False
        self._taskgroup = None
        # domain stats for AI suggestions
        self._domain_stats: Dict[str, Dict[str, int]] = {}
        # open output file
        self._out_fp = open(self.output_file, "a", encoding="utf-8")

    def enqueue(self, task: CrawlTask):
        """
        将任务放入队列：注意幂等/重复过滤，任务必须有明确 purpose
        """
        if task.url in self._visited:
            logger.debug(f"Skip enqueue visited: {task.url}")
            return
        # 这里不做严格去重（可为不同 purpose 重试），但默认简单去重
        self._task_queue.put_nowait(task)
        logger.debug(f"Enqueued task {task.purpose} {task.url}")

    async def _save_result(self, result: CrawlResult):
        """
        输出结构化结果：默认 JSONL，可替换为 DB/消息队列
        """
        record = {
            "url": result.url,
            "timestamp": result.timestamp,
            "status": result.status,
            "elapsed": result.elapsed,
            "data": result.data,
            "tx_hash": result.tx_hash
        }
        self._out_fp.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._out_fp.flush()
        logger.info(f"Saved result for {result.url}")

    def _update_domain_stats(self, domain: str, success: bool):
        st = self._domain_stats.setdefault(domain, {"successes": 0, "failures": 0})
        if success:
            st["successes"] += 1
        else:
            st["failures"] += 1

    # tenacity retry wrapper for network calls
    def network_retry_decorator(self):
        return retry(
            reraise=True,
            stop=stop_after_attempt(config.MAX_RETRIES),
            wait=wait_exponential(multiplier=config.RETRY_BASE, min=1, max=60),
            retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError))
        )

    async def _fetch(self, task: CrawlTask, use_proxy: bool = False) -> Tuple[int, str, float, Optional[Proxy]]:
        """
        内部 fetch：执行 HTTP 请求并返回 (status, text, elapsed, proxy_used)
        - 支持可选代理
        - 支持 circuit breaker 和 domain rate limiting
        """
        parsed = urlparse(task.url)
        domain = parsed.netloc

        # 检查 circuit breaker
        await self.cb.before(domain)

        # robots check
        allowed = await self.robots.allowed(task.url)
        if not allowed:
            logger.info(f"Disallowed by robots.txt: {task.url}")
            return 403, "", 0.0, None

        # rate limiter
        await self.rate_limiter.wait(domain)

        # optional proxy selection
        proxy_used: Optional[Proxy] = None
        if use_proxy and self.proxy_pool:
            try:
                proxy_used = await self.proxy_pool.get()
            except ProxyPoolEmpty:
                logger.warning("Proxy requested but pool empty; proceed without proxy")
                proxy_used = None

        @self.network_retry_decorator()
        async def _single_request():
            async with self._semaphore:
                session = await self.session_mgr.session()
                start = time.time()
                kwargs = {
                    "params": {},
                    "headers": {"User-Agent": config.USER_AGENT},
                    "timeout": aiohttp.ClientTimeout(total=config.DEFAULT_TIMEOUT),
                    "ssl": False
                }
                if proxy_used:
                    kwargs["proxy"] = proxy_used.url()
                async with session.request("GET", task.url, **kwargs) as resp:
                    text = await resp.text(errors="replace")
                    elapsed = time.time() - start
                    return resp.status, text, elapsed

        try:
            status, text, elapsed = await _single_request()
            # report proxy
            if proxy_used and self.proxy_pool:
                await self.proxy_pool.report(proxy_used, success=(200 <= status < 400))
            if PROMETHEUS_AVAILABLE:
                MET_REQUESTS.inc()
                if 200 <= status < 400:
                    MET_SUCCESS.inc()
                else:
                    MET_FAILED.inc()
            await self.cb.success(domain)
            self._update_domain_stats(domain, success=(200 <= status < 400))
            return status, text, elapsed, proxy_used
        except CircuitOpenError:
            logger.warning(f"Circuit open for {domain}, skip fetch {task.url}")
            raise
        except Exception as e:
            logger.error(f"Fetch error {task.url}: {e}")
            await self.cb.failure(domain)
            self._update_domain_stats(domain, success=False)
            if PROMETHEUS_AVAILABLE:
                MET_FAILED.inc()
            raise

    async def _process_task(self, task: CrawlTask):
        """
        处理单个任务：
        - 执行 fetch
        - 用 extractor 抽取结构化数据
        - 调用 AI 判断是否深入并决定是否生成子任务
        - 将结果写入存储
        """
        if task.url in self._visited:
            logger.debug(f"Already visited {task.url}")
            return
        self._visited.add(task.url)

        # choose whether to use proxy for this task (simple policy: use_proxy for untrusted domains)
        use_proxy = False  # policy placeholder

        try:
            status, text, elapsed, proxy_used = await self._fetch(task, use_proxy=use_proxy)
        except CircuitOpenError:
            return
        except Exception:
            return

        # extract structured data
        data = self.extractor.extract(text, task.url, task.purpose) if status == 200 else {}

        # optional blockchain proof of content (store hash)
        tx_hash = None
        if self.blockchain and status == 200:
            try:
                payload_hash = hashlib.sha256(json.dumps(data, ensure_ascii=False).encode()).hexdigest()
                tx_hash = await self.blockchain.store_crawl_proof(task.url, payload_hash)
            except Exception as e:
                logger.warning(f"Blockchain store failed for {task.url}: {e}")

        # save
        result = CrawlResult(
            url=task.url,
            timestamp=datetime.utcnow().isoformat(),
            status=status,
            elapsed=elapsed,
            data=data,
            tx_hash=tx_hash
        )
        await self._save_result(result)

        # AI decision: 是否继续深入（生成新任务）
        try:
            should = await self.ai.should_continue(text, task.url, {"purpose": task.purpose, "depth": task.depth})
        except Exception as e:
            logger.warning(f"AI scheduler error for should_continue: {e}")
            should = False

        if should and task.depth < task.meta.get("max_depth", 3):
            # extract candidate links and filter by rules (purpose-driven)
            links = self._extract_links(text, task.url)
            for link in links:
                # 简单过滤：只跟进相同域或白名单域；并保证不超过 max pages
                domain = urlparse(link).netloc
                # domain policy placeholder (可自定义)
                if task.meta.get("allowed_domains") and domain not in task.meta["allowed_domains"]:
                    continue
                # 进一步用 AI 或规则判断链接是否与目标相关
                related = False
                if task.purpose == "extract_article":
                    # quick heuristic: url or anchor contains keywords
                    for kw in task.meta.get("goal_keywords", []):
                        if kw.lower() in link.lower():
                            related = True
                            break
                else:
                    # default: be conservative — only follow same domain links
                    related = (domain == urlparse(task.url).netloc)

                if related:
                    new_task = CrawlTask(url=link, purpose=task.purpose, depth=task.depth + 1, meta=task.meta)
                    self.enqueue(new_task)

        # AI may also recommend adjusting domain interval
        try:
            domain_stats = self._domain_stats.get(urlparse(task.url).netloc, {})
            suggested = await self.ai.suggest_interval(urlparse(task.url).netloc, domain_stats)
            if suggested:
                self.rate_limiter.set_interval(urlparse(task.url).netloc, suggested)
        except Exception as e:
            logger.debug(f"AI scheduler suggest_interval error: {e}")

    def _extract_links(self, html: str, base_url: str) -> List[str]:
        """
        提取页面内的绝对链接（http/https），去 fragment。
        返回较干净的 URL 列表（不含 mailto/javascript）
        """
        try:
            soup = BeautifulSoup(html, "lxml")
        except Exception:
            soup = BeautifulSoup(html or "", "html.parser")
        links = set()
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if not href or href.startswith("#") or href.startswith("mailto:") or href.startswith("javascript:"):
                continue
            if href.startswith("http://") or href.startswith("https://"):
                abs_url = href.split("#")[0]
            else:
                abs_url = urljoin(base_url, href).split("#")[0]
            if abs_url.startswith("http://") or abs_url.startswith("https://"):
                links.add(abs_url)
        return list(links)

    async def _worker(self):
        """
        worker loop:拉任务 -> 处理 -> 循环，直到 stop 请求
        """
        while not self._stop_requested:
            try:
                task: CrawlTask = await asyncio.wait_for(self._task_queue.get(), timeout=2.0)
            except asyncio.TimeoutError:
                # 否则继续等待直到 stop
                continue
            try:
                await self._process_task(task)
            except Exception as e:
                logger.exception(f"Error processing task {task.url}: {e}")
            finally:
                self._task_queue.task_done()

    async def start(self, initial_tasks: List[CrawlTask], workers: int = 8):
        """
        启动内核：
        - 向队列注入初始任务
        - 创建 worker TaskGroup
        """
        if workers < 1:
            workers = 1
        for t in initial_tasks:
            self.enqueue(t)
        # run workers within TaskGroup for cancellation management
        async with asyncio.TaskGroup() as tg:
            self._taskgroup = tg
            for _ in range(workers):
                tg.create_task(self._worker())
            # TaskGroup will wait until all tasks complete; to stop externally, call stop()
        logger.info("CoreCrawler stopped TaskGroup scope")

    async def stop(self):
        """
        请求停止：设置标志并等待队列耗尽或强制退出
        """
        self._stop_requested = True
        logger.info("Stop requested for CoreCrawler")
        # 等待队列耗尽
        await self._task_queue.join()
        # 关闭输出
        self._out_fp.flush()
        self._out_fp.close()

    async def close(self):
        await self.session_mgr.close()

# ---------------------------
# Graceful signal wiring utility
# ---------------------------
def install_signal_handlers(crawler: CoreCrawler):
    loop = asyncio.get_event_loop()

    def _handler(sig):
        logger.info(f"Signal received: {sig}. Requesting crawler stop...")
        asyncio.create_task(crawler.stop())

    for s in ("SIGINT", "SIGTERM"):
        if hasattr(signal, s):
            loop.add_signal_handler(getattr(signal, s), lambda s=s: _handler(s))

# ---------------------------
# Example usage (示范如何用核心)，请在实际部署时替换占位实现
# ---------------------------
async def main():
    # 初始化组件
    session_mgr = SessionManager()
    robots = RobotsCache()
    rate_limiter = DomainRateLimiter()
    cb = CircuitBreaker()
    extractor = StructuredExtractor()

    # AI：使用启发式内置实现，也可以传入 LLM_AIScheduler(llm_call=...) 来调用外部 LLM
    ai_scheduler = HeuristicAIScheduler(min_len=400, important_words=["article", "news", "price", "product"])

    # 可选代理池
    proxy_pool = ProxyPool(session_mgr)
    # 示例：从环境或文件注入代理（格式 ip:port:protocol）
    env_proxies = os.getenv("INITIAL_PROXIES")
    if env_proxies:
        specs = []
        for p in env_proxies.split(","):
            try:
                ip, port, proto = p.split(":")
                specs.append({"ip": ip, "port": int(port), "protocol": proto})
            except Exception:
                continue
        if specs:
            await proxy_pool.add(specs)

    # 区块链占位集成（可替换）
    blockchain = BlockchainIntegration()

    # 内核实例
    crawler = CoreCrawler(
        session_mgr=session_mgr,
        robots=robots,
        rate_limiter=rate_limiter,
        cb=cb,
        extractor=extractor,
        ai_scheduler=ai_scheduler,
        proxy_pool=proxy_pool,
        blockchain=blockchain
    )

    install_signal_handlers(crawler)

    # 构造初始任务（明确目的）
    initial = [
        CrawlTask(url="https://example.com/", purpose="extract_article", depth=0,
                  meta={"goal_keywords": ["example", "news", "article"], "allowed_domains": ["example.com"], "max_depth": 2})
    ]

    try:
        # 启动内核（并发 worker 数量可配置）
        await crawler.start(initial, workers=6)
    finally:
        await crawler.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Terminated by user")

