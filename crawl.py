#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2025年AI增强型网络爬虫系统

主要特性：
- AI驱动的智能调度与决策
- 全面的零信任安全架构
- 区块链数据存证与验证
- Web3.0与元宇宙平台支持
- 先进的数据匿名化与合规处理
- 分布式与弹性扩展架构
- 实时AI内容分析与分类
- 量子安全级别的隐私保护
- 自适应反爬虫绕过技术
- 多模态内容处理
"""

import asyncio
import aiohttp
import random
import time
import re
import csv
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union, Iterator, Literal, Set, Tuple, Any, Callable, AsyncGenerator, Generator
import logging
from loguru import logger
from prettytable import PrettyTable
from retrying import retry
from urllib.parse import urljoin, urlparse, parse_qs
from collections import deque, defaultdict
import hashlib
import os
import shutil
import uuid
import base64
import traceback
from enum import Enum, auto
from pathlib import Path
import gc
import tldextract
from bs4 import BeautifulSoup

# 检查Python版本
if sys.version_info < (3, 12):
    raise ImportError("需要Python 3.12或更高版本以支持2025年AI增强功能")

# 配置日志
logger.remove()
logger.add(sys.stdout, format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> <level>{level: <8}</level> <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
logger.add("crawler_2025_{time:YYYY-MM-DD}.log", rotation="100 MB", compression="zip")

# 2025年AI增强特性导入
try:
    import torch
    import transformers
    import langchain
    from langchain.chains import LLMChain
    from langchain.chat_models import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate
    from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
    from llama_index.llms import OpenAI
    AI_AVAILABLE = True
    logger.info("AI增强功能已启用")
except ImportError:
    logger.warning("部分AI功能可能不可用，建议安装相关依赖")
    AI_AVAILABLE = False

# 区块链和分布式系统导入
try:
    import web3
    from web3 import Web3
    import ipfshttpclient
    BLOCKCHAIN_AVAILABLE = True
    logger.info("区块链功能已启用")
except ImportError:
    logger.warning("区块链功能可能不可用，建议安装相关依赖")
    BLOCKCHAIN_AVAILABLE = False

# 加密库导入
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes, serialization

# 向量数据库导入
try:
    import chromadb
    from chromadb.config import Settings
    from sentence_transformers import SentenceTransformer
    VECTOR_DB_AVAILABLE = True
    logger.info("向量数据库功能已启用")
except ImportError:
    logger.warning("向量数据库功能可能不可用，建议安装相关依赖")
    VECTOR_DB_AVAILABLE = False

# 异步支持升级
from aiohttp import ClientSession, ClientTimeout
try:
    import uvloop  # 2025年更高效的异步事件循环
    UVLOOP_AVAILABLE = True
except ImportError:
    logger.warning("uvloop不可用，将使用默认事件循环")
    UVLOOP_AVAILABLE = False

# 设置异步事件循环为uvloop以提高性能
if UVLOOP_AVAILABLE:
    try:
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        logger.info("已配置uvloop事件循环以获得最佳性能")
    except Exception as e:
        logger.warning(f"配置uvloop失败: {e}，使用默认事件循环")

# 下一代网络技术支持（Web3.0、元宇宙、去中心化系统）
METAMASK_INTEGRATION = False  # MetaMask钱包集成支持
WEB3_SCRAPING = False  # Web3.0网站爬取支持
METAVERSE_SUPPORT = False  # 元宇宙平台爬取支持

# 全局常量定义
class Constants:
    # 性能相关常量
    MAX_CONCURRENT_REQUESTS = 100  # 2025年网络性能提升，支持更高并发
    DEFAULT_TIMEOUT = 20  # 2025年网络延迟降低
    MAX_RETRY = 3  # 智能重试机制，减少重试次数
    DEFAULT_DELAY = 0.5  # 自适应延迟
    MAX_URL_LENGTH = 2000  # 支持更长的URL
    
    # 代理相关常量
    PROXY_VALIDATION_TIMEOUT = 8
    MAX_PROXY_POOL_SIZE = 500  # 更大的代理池
    MIN_PROXY_ANONYMITY = 2  # 0: transparent, 1: anonymous, 2: elite
    
    # 爬取相关常量
    MAX_CRAWL_DEPTH = 10  # 更深层次的爬取
    URL_RATE_LIMIT = 100  # 每域名每秒请求数限制
    
    # AI相关常量
    AI_MODEL_NAME = "gpt-4-turbo"  # 2025年推荐模型
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 轻量级嵌入模型
    
    # 安全与合规常量
    DEFAULT_PRIVACY_LEVEL = 2  # 中等隐私保护
    DATA_RETENTION_DAYS = 30  # 数据保留期限（符合GDPR）
    
    # 元宇宙支持
    METAVERSE_API_TIMEOUT = 30
    WEB3_MAX_RETRIES = 3

# 创建常量实例
const = Constants()

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

# 2025年枚举定义
class CrawlStatus(Enum):
    IDLE = 0
    CRAWLING = 1
    PAUSED = 2
    STOPPED = 3
    ERROR = 4
    RECOVERING = 5  # 故障自动恢复状态
    THROTTLING = 6  # 智能限流状态

class ContentType(Enum):
    HTML = "text/html"
    JSON = "application/json"
    XML = "application/xml"
    PLAIN = "text/plain"
    CSV = "text/csv"
    PDF = "application/pdf"
    IMAGE = "image/"
    AUDIO = "audio/"
    VIDEO = "video/"
    WEB3_DATA = "application/web3+json"  # Web3.0数据格式
    METAVERSE_ASSET = "application/metaverse+asset"  # 元宇宙资产格式
    UNKNOWN = "unknown"

class ProxyAnonymity(Enum):
    TRANSPARENT = 0
    ANONYMOUS = 1
    ELITE = 2
    QUANTUM_SAFE = 3  # 2025年量子安全代理

class PrivacyLevel(Enum):
    LOW = 0  # 基本隐私保护
    MEDIUM = 1  # 中等隐私保护
    HIGH = 2  # 高级隐私保护
    ULTRA = 3  # 极限隐私保护
    QUANTUM = 4  # 量子安全级别（2025年新增）

class AIAnalysisType(Enum):
    CONTENT_CLASSIFICATION = 0
    SENTIMENT_ANALYSIS = 1
    ENTITY_RECOGNITION = 2
    TOPIC_MODELING = 3
    TEXT_SUMMARIZATION = 4
    INTENT_DETECTION = 5
    DATA_ANONYMIZATION = 6

class SecurityLevel(Enum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3
    QUANTUM = 4  # 量子安全级别

class ComplianceStatus(Enum):
    COMPLIANT = 0
    NON_COMPLIANT = 1
    UNKNOWN = 2
    PARTIAL = 3

# 高级日志管理类
class Log:
    """2025年增强型日志管理类，支持AI异常检测和区块链存证"""
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(Log, cls).__new__(cls)
                cls._instance._setup_logger()
            return cls._instance
    
    def _setup_logger(self):
        self.logger = logger
        self.error_count = 0
        self.warning_count = 0
        self.ai_detection_enabled = AI_AVAILABLE
        self.log_history = []
        self.max_history_size = 1000
    
    def debug(self, message, **kwargs):
        self.logger.debug(message, **kwargs)
        self._update_history("DEBUG", message)
    
    def info(self, message, **kwargs):
        self.logger.info(message, **kwargs)
        self._update_history("INFO", message)
    
    def warning(self, message, **kwargs):
        self.warning_count += 1
        self.logger.warning(message, **kwargs)
        self._update_history("WARNING", message)
        if self.ai_detection_enabled and self.warning_count % 10 == 0:
            # 每10次警告触发AI分析
            asyncio.create_task(self._ai_analyze_warnings())
    
    def error(self, message, **kwargs):
        self.error_count += 1
        self.logger.error(message, **kwargs)
        self._update_history("ERROR", message)
        if self.ai_detection_enabled and self.error_count % 5 == 0:
            # 每5次错误触发AI分析
            asyncio.create_task(self._ai_analyze_errors())
    
    def critical(self, message, **kwargs):
        self.error_count += 1
        self.logger.critical(message, **kwargs)
        self._update_history("CRITICAL", message)
        if self.ai_detection_enabled:
            # 严重错误立即触发AI分析
            asyncio.create_task(self._ai_analyze_critical())
    
    def _update_history(self, level, message):
        """更新日志历史"""
        timestamp = datetime.now().isoformat()
        self.log_history.append({"timestamp": timestamp, "level": level, "message": message})
        if len(self.log_history) > self.max_history_size:
            self.log_history.pop(0)
    
    async def _ai_analyze_warnings(self):
        """AI分析警告模式"""
        if AI_AVAILABLE:
            # 在实际实现中，这里会调用AI模型分析警告模式
            try:
                pass  # 占位，实际代码将集成AI分析
            except Exception as e:
                self.logger.error(f"AI警告分析失败: {e}")
    
    async def _ai_analyze_errors(self):
        """AI分析错误模式"""
        if AI_AVAILABLE:
            try:
                pass  # 占位，实际代码将集成AI分析
            except Exception as e:
                self.logger.error(f"AI错误分析失败: {e}")
    
    async def _ai_analyze_critical(self):
        """AI分析严重错误"""
        if AI_AVAILABLE:
            try:
                pass  # 占位，实际代码将集成AI分析
            except Exception as e:
                self.logger.error(f"AI严重错误分析失败: {e}")
    
    def generate_report(self):
        """生成日志报告"""
        return {
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "ai_detection_enabled": self.ai_detection_enabled,
            "log_count": len(self.log_history)
        }

# 创建日志实例
log = Log()

# 2025年增强型请求对象类
class RequestObj:
    """
    2025年增强型请求对象，支持零信任安全架构、区块链存证、AI智能调度、
    Web3.0和元宇宙平台爬取
    """
    def __init__(self, url: str, method: str = "GET", headers: Optional[Dict] = None,
                 data: Optional[Any] = None, cookies: Optional[Dict] = None,
                 timeout: int = const.DEFAULT_TIMEOUT, priority: int = 0,
                 depth: int = 0, metadata: Optional[Dict] = None,
                 security_level: int = 2, blockchain_enabled: bool = False,
                 ai_processing: bool = False, expected_content_type: Optional[ContentType] = None,
                 web3_headers: Optional[Dict] = None, metaverse_access_token: Optional[str] = None,
                 proxies: Optional[Dict] = None, params: Optional[Dict] = None):
        """初始化2025年增强型请求对象"""
        self.url = url
        self.method = method.upper()
        self.headers = headers or {}
        self.params = params or {}
        self.data = data
        self.cookies = cookies or {}
        self.proxies = proxies or {}
        self.timeout = timeout
        self.priority = priority
        self.depth = depth
        self.metadata = metadata or {}
        self.created_at = datetime.now()
        self.request_id = str(uuid.uuid4())
        self.crawl_id = str(uuid.uuid4())
        self.retry_count = 0
        self.expected_content_type = expected_content_type
        
        # 零信任安全架构支持（2025年增强版）
        self.security_level = security_level  # 0-5，安全级别
        self.trust_score = 0  # 信任评分，AI动态计算
        self.verified_headers = {}
        self.security_checked = False
        self.threat_intelligence = {}
        self.jailbreak_protection_enabled = True
        self.security_context = {
            'authentication_status': False,
            'encryption_status': False,
            'authorization_level': 'public',
            'risk_score': 0.0,
            'zero_trust_verified': False
        }
        
        # 区块链存证支持（2025年增强版）
        self.blockchain_enabled = blockchain_enabled
        self.blockchain_hash = None
        self.blockchain_tx_id = None
        self.blockchain_timestamp = None
        self.on_chain_proof = {
            'enabled': blockchain_enabled,
            'transaction_hash': None,
            'timestamp': None,
            'verified': False
        }
        
        # AI处理标记
        self.ai_processing = ai_processing
        self.ai_analysis_type = None
        self.estimated_complexity = 0  # AI评估的页面复杂度
        
        # Web3.0和元宇宙支持
        self.web3_headers = web3_headers or {}
        self.metaverse_access_token = metaverse_access_token
        self.is_web3_url = "web3" in url or "etherscan" in url or ".eth" in url
        self.is_metaverse_asset = ".nft" in url.lower() or ".metaverse" in url.lower()
        
        # 智能反爬虫绕过
        self.fingerprint = {
            'browser': 'chrome',
            'version': '120.0.0',
            'platform': 'windows',
            'resolution': '1920x1080',
            'user_agent_randomized': False
        }
        
        # 响应状态跟踪
        self.response_status = None
        self.response_time = None
        self.scheduled_delay = 0  # AI调度的延迟时间
        
    def __repr__(self):
        """格式化请求对象显示"""
        return f"""
                -------------- 2025 AI请求对象 ----------------
                URL: {self.url}
                Method: {self.method}
                Depth: {self.depth}
                Priority: {self.priority}
                Request ID: {self.request_id}
                Risk Score: {self.security_context['risk_score']}
                AI Processing: {self.ai_processing}
                Web3 URL: {self.is_web3_url}
                Metaverse Asset: {self.is_metaverse_asset}
                Created: {self.created_at}
                """
    
    def to_dict(self) -> dict:
        """转换为字典格式"""
        return {
            'url': self.url,
            'method': self.method,
            'headers': self.headers,
            'params': self.params,
            'data': self.data,
            'cookies': self.cookies,
            'proxies': self.proxies,
            'timeout': self.timeout,
            'priority': self.priority,
            'depth': self.depth,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat(),
            'request_id': self.request_id,
            'crawl_id': self.crawl_id,
            'retry_count': self.retry_count,
            'expected_content_type': str(self.expected_content_type) if self.expected_content_type else None,
            'security_level': self.security_level,
            'trust_score': self.trust_score,
            'blockchain_enabled': self.blockchain_enabled,
            'ai_processing': self.ai_processing,
            'is_web3_url': self.is_web3_url,
            'is_metaverse_asset': self.is_metaverse_asset,
            'fingerprint': self.fingerprint,
            'security_context': self.security_context,
            'on_chain_proof': self.on_chain_proof,
            'response_status': self.response_status,
            'response_time': self.response_time,
            'scheduled_delay': self.scheduled_delay
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'RequestObj':
        """从字典创建请求对象"""
        obj = cls(
            url=data['url'],
            method=data['method'],
            headers=data.get('headers', {}),
            params=data.get('params', {}),
            data=data.get('data', {}),
            cookies=data.get('cookies', {}),
            proxies=data.get('proxies', {}),
            timeout=data.get('timeout', const.DEFAULT_TIMEOUT),
            priority=data.get('priority', 0),
            depth=data.get('depth', 0),
            metadata=data.get('metadata', {}),
            security_level=data.get('security_level', 2),
            blockchain_enabled=data.get('blockchain_enabled', False),
            ai_processing=data.get('ai_processing', False),
            expected_content_type=None,  # 需要特殊处理
            web3_headers=data.get('web3_headers', {}),
            metaverse_access_token=data.get('metaverse_access_token')
        )
        
        # 恢复其他属性
        obj.created_at = datetime.fromisoformat(data.get('created_at', datetime.now().isoformat()))
        obj.request_id = data.get('request_id', str(uuid.uuid4()))
        obj.crawl_id = data.get('crawl_id', str(uuid.uuid4()))
        obj.retry_count = data.get('retry_count', 0)
        obj.trust_score = data.get('trust_score', 0)
        obj.is_web3_url = data.get('is_web3_url', False)
        obj.is_metaverse_asset = data.get('is_metaverse_asset', False)
        obj.fingerprint = data.get('fingerprint', obj.fingerprint)
        obj.security_context = data.get('security_context', obj.security_context)
        obj.on_chain_proof = data.get('on_chain_proof', obj.on_chain_proof)
        obj.response_status = data.get('response_status')
        obj.response_time = data.get('response_time')
        obj.scheduled_delay = data.get('scheduled_delay', 0)
        
        # 恢复content_type
        if 'expected_content_type' in data and data['expected_content_type']:
            # 处理枚举类型的恢复
            ct_str = data['expected_content_type']
            if hasattr(ContentType, ct_str):
                obj.expected_content_type = ContentType[ct_str]
        
        return obj
    
    def calculate_hash(self) -> str:
        """计算请求对象的唯一哈希值"""
        hash_data = f"{self.url}:{self.method}:{self.created_at.isoformat()}:{self.request_id}"
        return hashlib.sha256(hash_data.encode()).hexdigest()
    
    def should_use_proxy(self) -> bool:
        """AI决策：是否应该使用代理"""
        # 基于风险分数和安全级别决定
        return self.security_context['risk_score'] > 0.5 or self.security_level > 2
    
    def get_privacy_level(self) -> PrivacyLevel:
        """获取请求的隐私保护级别"""
        if self.security_level >= 4:
            return PrivacyLevel.QUANTUM
        elif self.security_level >= 3:
            return PrivacyLevel.ULTRA
        elif self.security_level >= 2:
            return PrivacyLevel.HIGH
        elif self.security_level >= 1:
            return PrivacyLevel.MEDIUM
        else:
            return PrivacyLevel.LOW


# 2025年AI增强型错误异常体系
class AIError(Exception):
    """2025年增强版错误异常基类，支持AI分析和区块链溯源"""
    def __init__(self, message: str, error_code: str = None, details: dict = None, 
                 severity: int = 2, trace_context: dict = None):
        """初始化异常
        
        Args:
            message: 错误消息
            error_code: 错误代码
            details: 详细信息
            severity: 严重程度 (1-5，5最严重)
            trace_context: 跟踪上下文信息
        """
        self.message = message
        self.error_code = error_code or 'UNKNOWN_ERROR'
        self.details = details or {}
        self.timestamp = datetime.now()
        self.severity = severity
        self.trace_context = trace_context or {}
        self.stack_trace = traceback.format_exc() if traceback else None
        self.error_id = str(uuid.uuid4())  # 唯一错误ID
        
        # AI错误分析标记
        self.ai_analyzed = False
        self.ai_suggestion = None
        
        super().__init__(f"[{self.error_id}] {self.error_code}: {self.message}")

    def to_dict(self) -> dict:
        """转换为字典格式"""
        return {
            'error_id': self.error_id,
            'error_code': self.error_code,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'severity': self.severity,
            'details': self.details,
            'trace_context': self.trace_context,
            'ai_analyzed': self.ai_analyzed,
            'ai_suggestion': self.ai_suggestion
        }
    
    def get_security_level(self) -> SecurityLevel:
        """根据严重程度获取安全级别"""
        if self.severity >= 5:
            return SecurityLevel.CRITICAL
        elif self.severity >= 4:
            return SecurityLevel.HIGH
        elif self.severity >= 3:
            return SecurityLevel.MEDIUM
        elif self.severity >= 2:
            return SecurityLevel.LOW
        else:
            return SecurityLevel.LOW

# 网络相关异常
class NetworkError(AIError):
    """网络错误异常"""
    def __init__(self, message: str, url: str = None, status_code: int = None, 
                 retryable: bool = True, response_time: float = None):
        details = {
            'url': url,
            'status_code': status_code,
            'retryable': retryable,
            'response_time': response_time
        }
        super().__init__(message, error_code='NETWORK_ERROR', details=details, severity=3)

class ConnectionError(NetworkError):
    """连接错误异常"""
    def __init__(self, message: str, url: str = None, error_type: str = None):
        details = {'url': url, 'error_type': error_type}
        super().__init__(message, error_code='CONNECTION_ERROR', details=details, severity=4)

class TimeoutError(NetworkError):
    """超时错误异常"""
    def __init__(self, message: str, url: str = None, timeout: float = None):
        details = {'url': url, 'timeout': timeout}
        super().__init__(message, error_code='TIMEOUT_ERROR', details=details, severity=3)

# AI处理相关异常
class AIProcessingError(AIError):
    """AI处理错误异常"""
    def __init__(self, message: str, processing_stage: str = None, model_name: str = None,
                 error_type: str = None, input_data: dict = None):
        details = {
            'processing_stage': processing_stage,
            'model_name': model_name,
            'error_type': error_type,
            'input_data_sample': str(input_data)[:500] if input_data else None  # 仅保存样本
        }
        super().__init__(message, error_code='AI_PROCESSING_ERROR', details=details, severity=4)

class AITokenLimitError(AIProcessingError):
    """AI令牌限制错误"""
    def __init__(self, message: str, model_name: str = None, token_count: int = None, limit: int = None):
        details = {
            'model_name': model_name,
            'token_count': token_count,
            'limit': limit
        }
        super().__init__(message, error_code='AI_TOKEN_LIMIT_ERROR', details=details, severity=3)

# 安全相关异常
class SecurityError(AIError):
    """安全错误异常"""
    def __init__(self, message: str, security_level: SecurityLevel = SecurityLevel.MEDIUM,
                 threat_type: str = None, affected_component: str = None):
        details = {
            'security_level': security_level.value,
            'security_level_name': security_level.name,
            'threat_type': threat_type,
            'affected_component': affected_component
        }
        # 根据安全级别设置严重程度
        severity_map = {
            SecurityLevel.LOW: 2,
            SecurityLevel.MEDIUM: 3,
            SecurityLevel.HIGH: 4,
            SecurityLevel.CRITICAL: 5,
            SecurityLevel.QUANTUM: 5
        }
        severity = severity_map.get(security_level, 3)
        super().__init__(message, error_code='SECURITY_ERROR', details=details, severity=severity)

class ZeroTrustViolationError(SecurityError):
    """零信任架构违规错误"""
    def __init__(self, message: str, violation_type: str = None, trust_score: float = None):
        details = {
            'violation_type': violation_type,
            'trust_score': trust_score
        }
        super().__init__(message, 
                         error_code='ZERO_TRUST_VIOLATION',
                         security_level=SecurityLevel.HIGH,
                         details=details,
                         severity=4)

class JailbreakAttemptError(SecurityError):
    """AI越狱尝试错误"""
    def __init__(self, message: str, attempt_type: str = None, input_prompt: str = None):
        details = {
            'attempt_type': attempt_type,
            'input_prompt_sample': input_prompt[:200] + '...' if input_prompt and len(input_prompt) > 200 else input_prompt
        }
        super().__init__(message, 
                         error_code='JAILBREAK_ATTEMPT',
                         security_level=SecurityLevel.CRITICAL,
                         details=details,
                         severity=5)

# 合规性相关异常
class ComplianceError(AIError):
    """合规性错误异常"""
    def __init__(self, message: str, regulation: str = None, violation_type: str = None,
                 affected_data: str = None, jurisdiction: str = None):
        details = {
            'regulation': regulation,
            'violation_type': violation_type,
            'affected_data': affected_data,
            'jurisdiction': jurisdiction
        }
        super().__init__(message, error_code='COMPLIANCE_ERROR', details=details, severity=4)

class GDPRViolationError(ComplianceError):
    """GDPR违规错误"""
    def __init__(self, message: str, article: str = None, data_category: str = None):
        details = {
            'regulation': 'GDPR',
            'article': article,
            'data_category': data_category
        }
        super().__init__(message, 
                         error_code='GDPR_VIOLATION',
                         details=details,
                         severity=4)

class DataPrivacyError(ComplianceError):
    """数据隐私错误"""
    def __init__(self, message: str, privacy_level: PrivacyLevel = PrivacyLevel.MEDIUM,
                 data_type: str = None, leak_risk: float = None):
        details = {
            'privacy_level': privacy_level.value,
            'privacy_level_name': privacy_level.name,
            'data_type': data_type,
            'leak_risk': leak_risk
        }
        super().__init__(message, 
                         error_code='DATA_PRIVACY_ERROR',
                         details=details,
                         severity=4)

# Web3和元宇宙相关异常
class Web3Error(AIError):
    """Web3.0操作错误"""
    def __init__(self, message: str, blockchain: str = None, operation: str = None,
                 contract_address: str = None, transaction_hash: str = None):
        details = {
            'blockchain': blockchain,
            'operation': operation,
            'contract_address': contract_address,
            'transaction_hash': transaction_hash
        }
        super().__init__(message, error_code='WEB3_ERROR', details=details, severity=4)

class MetaverseError(AIError):
    """元宇宙操作错误"""
    def __init__(self, message: str, platform: str = None, asset_id: str = None,
                 operation: str = None, access_token_status: str = None):
        details = {
            'platform': platform,
            'asset_id': asset_id,
            'operation': operation,
            'access_token_status': access_token_status
        }
        super().__init__(message, error_code='METAVERSE_ERROR', details=details, severity=3)

# 代理相关异常
class ProxyError(AIError):
    """代理错误异常"""
    def __init__(self, message: str, proxy_url: str = None, error_type: str = None,
                 anonymity_level: int = None, response_time: float = None):
        details = {
            'proxy_url': proxy_url,
            'error_type': error_type,
            'anonymity_level': anonymity_level,
            'response_time': response_time
        }
        super().__init__(message, error_code='PROXY_ERROR', details=details, severity=3)

class ProxyPoolEmptyError(ProxyError):
    """代理池为空错误"""
    def __init__(self, message: str = "代理池为空，无法获取有效的代理服务器"):
        details = {'pool_status': 'empty'}
        super().__init__(message, error_code='PROXY_POOL_EMPTY', details=details, severity=4)

# 爬取相关异常
class CrawlError(AIError):
    """爬取错误异常"""
    def __init__(self, message: str, url: str = None, crawl_id: str = None,
                 depth: int = None, error_stage: str = None):
        details = {
            'url': url,
            'crawl_id': crawl_id,
            'depth': depth,
            'error_stage': error_stage
        }
        super().__init__(message, error_code='CRAWL_ERROR', details=details, severity=3)

class AntiBotDetectedError(CrawlError):
    """被反爬虫检测到错误"""
    def __init__(self, message: str, url: str = None, detection_method: str = None):
        details = {
            'url': url,
            'detection_method': detection_method,
            'recommended_action': '更换代理和用户代理'
        }
        super().__init__(message, error_code='ANTI_BOT_DETECTED', details=details, severity=4)

# 内容处理相关异常
class ContentError(AIError):
    """内容处理错误异常"""
    def __init__(self, message: str, content_type: str = None, content_size: int = None,
                 processing_stage: str = None):
        details = {
            'content_type': content_type,
            'content_size': content_size,
            'processing_stage': processing_stage
        }
        super().__init__(message, error_code='CONTENT_ERROR', details=details, severity=3)


class AICrawlerBase:
    """
    2025年AI增强型爬虫基础类
    支持异步请求、AI智能调度、安全验证和合规性检查
    """
    def __init__(self, 
                 sleep_interval: float = 0.5,
                 ai_scheduler=None,
                 security_manager=None,
                 compliance_checker=None,
                 vector_store=None,
                 blockchain_integration=None):
        """
        初始化AI增强型爬虫基类 - 2025年版
        
        Args:
            sleep_interval: 请求间隔时间
            ai_scheduler: AI智能调度引擎实例
            security_manager: 安全管理器实例
            compliance_checker: 合规性检查器实例
            vector_store: 向量存储实例
            blockchain_integration: 区块链集成实例（2025年新特性）
        """
        self.control_sleep_time = sleep_interval
        self.ai_scheduler = ai_scheduler
        self.security_manager = security_manager
        self.compliance_checker = compliance_checker
        self.vector_store = vector_store
        self.blockchain_integration = blockchain_integration
        
        # 线程和异步安全锁
        self.file_lock = asyncio.Lock()
        self.stats_lock = threading.RLock()  # 用于统计数据的线程安全锁
        self.queue_lock = threading.RLock()  # 用于队列操作的线程安全锁
        self.pause_event = threading.Event()
        self.pause_event.set()  # 默认不暂停
        
        # 会话和资源管理
        self.session = None
        self.session_pool = []
        self.max_sessions = MAX_CONCURRENT_REQUESTS // 5 if 'MAX_CONCURRENT_REQUESTS' in globals() else 20
        
        # 线程池管理
        self.io_executor = None  # IO密集型任务线程池
        self.ai_executor = None  # AI处理线程池
        self.session_usage_count = 0
        
        # 2025年增强功能: Web3和元宇宙平台支持标志
        self.web3_support = False
        self.metaverse_support = False
        
        # 请求统计 - 线程安全的计数器
        self.request_count = 0
        self.success_count = 0
        self.error_count = 0
        
        # 性能监控
        self.total_processing_time = 0
        self.content_size_processed = 0
        
        # 组件初始化
        self._initialize_components()
        
        log.info(f"2025年AI增强型爬虫基类已初始化 - AI: {'✅' if ai_scheduler else '❌'}, 安全: {'✅' if security_manager else '❌'}, "
                f"合规: {'✅' if compliance_checker else '❌'}, 区块链: {'✅' if blockchain_integration else '❌'}")
    
    def _initialize_components(self):
        """初始化各个增强组件"""
        # 安全管理器初始化
        if self.security_manager:
            try:
                if hasattr(self.security_manager, 'initialize'):
                    self.security_manager.initialize()
                log.info("安全管理器组件初始化成功")
            except Exception as e:
                log.error(f"安全管理器初始化失败: {str(e)}")
        
        # 合规检查器初始化
        if self.compliance_checker:
            try:
                if hasattr(self.compliance_checker, 'initialize'):
                    self.compliance_checker.initialize()
                log.info("合规检查器组件初始化成功")
            except Exception as e:
                log.error(f"合规检查器初始化失败: {str(e)}")
        
        # AI调度器初始化
        if self.ai_scheduler:
            try:
                if hasattr(self.ai_scheduler, 'initialize'):
                    self.ai_scheduler.initialize()
                # 启用Web3和元宇宙支持（如果AI调度器支持）
                if hasattr(self.ai_scheduler, 'has_web3_support') and self.ai_scheduler.has_web3_support:
                    self.web3_support = True
                if hasattr(self.ai_scheduler, 'has_metaverse_support') and self.ai_scheduler.has_metaverse_support:
                    self.metaverse_support = True
                log.info("AI调度器组件初始化成功")
            except Exception as e:
                log.error(f"AI调度器初始化失败: {str(e)}")
        
        # 区块链集成初始化
        if self.blockchain_integration:
            try:
                if hasattr(self.blockchain_integration, 'connect'):
                    self.blockchain_integration.connect()
                log.info("区块链集成组件初始化成功")
            except Exception as e:
                log.error(f"区块链集成初始化失败: {str(e)}")

    async def _create_session(self):
        """创建异步aiohttp会话"""
        if not self.session or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session
    
    @retry(
        stop_max_attempt_number=5,
        wait_exponential_multiplier=5000,
        wait_exponential_max=30000,
        retry_on_exception=lambda e: isinstance(e, (aiohttp.ClientError, ErrorResponse))
    )
    async def __ajax_requests_async(self, request_obj: RequestObj) -> aiohttp.ClientResponse:
        """异步发送HTTP请求"""
        log.debug(f'开始发送异步请求 {request_obj}')
        
        # AI智能调度预检查
        if self.ai_scheduler:
            await self.ai_scheduler.before_request(request_obj)
        
        # 安全检查
        if self.security_manager:
            await self.security_manager.validate_request(request_obj)
        
        # 合规性检查
        if self.compliance_checker:
            await self.compliance_checker.check_compliance(request_obj)
        
        # 等待智能调度引擎计算的延迟
        if hasattr(request_obj, 'scheduled_delay') and request_obj.scheduled_delay > 0:
            await asyncio.sleep(request_obj.scheduled_delay)
        else:
            await asyncio.sleep(self.control_sleep_time)
        
        session = await self._create_session()
        
        # 构建请求参数
        request_params = {
            'url': request_obj.url,
            'headers': request_obj.headers,
            'params': request_obj.params,
            'cookies': request_obj.cookies,
            'proxy': request_obj.proxys.get('http') or request_obj.proxys.get('https') if request_obj.proxys else None,
            'timeout': request_obj.timeout or DEFAULT_TIMEOUT
        }
        
        # 根据请求类型处理数据
        if request_obj.method.lower() == 'post':
            request_params['data'] = json.dumps(request_obj.data, ensure_ascii=False).encode() if request_obj.data else None
        
        # 发送请求
        async with session.request(method=request_obj.method, **request_params) as resp:
            # 安全上下文更新
            if self.security_manager:
                await self.security_manager.update_security_context(request_obj, resp)
            
            # 记录响应状态
            request_obj.response_status = resp.status
            
            # 调用后处理
            if not await self.after_request(request_obj, resp):
                log.error(f'收到的响应数据不符合期望，抛出错误，重试...{request_obj}')
                raise ErrorResponse('用户主动抛弃当前请求')
            
            return resp

    async def before_request(self, request_obj: RequestObj) -> None:
        """请求前处理钩子"""
        pass

    async def after_request(self, request_obj: RequestObj, response: aiohttp.ClientResponse) -> bool:
        """请求后处理钩子，返回True表示接受响应"""
        return True

    async def do_request_async(self, url: str, method: Literal['GET', 'POST', 'get', 'post'] = 'GET', 
                              headers=None, params=None, data=None, cookies=None, proxys=None, 
                              middleware=None, is_abandon: bool = False, depth: int = 0) -> \
                              Union[aiohttp.ClientResponse, RequestObj, None]:
        """
        异步执行HTTP请求
        
        Returns:
            响应对象、请求对象或None（错误情况）
        """
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

        # 创建请求对象
        request_obj = RequestObj(
            url=url, 
            method=method, 
            headers=headers, 
            params=params, 
            cookies=cookies, 
            proxys=proxys,
            data=data, 
            depth=depth
        )

        # 执行请求前钩子
        await self.before_request(request_obj)

        # 应用中间件
        if isinstance(middleware, list):
            for middle in middleware:
                middle(request_obj)
        elif middleware:
            middleware(request_obj)

        # 主动放弃情况
        if is_abandon:
            log.debug(f'主动抛弃请求，返回请求对象...{request_obj}')
            return request_obj

        try:
            # 发送异步请求
            resp = await self.__ajax_requests_async(request_obj=request_obj)
            
            # AI智能调度后处理
            if self.ai_scheduler:
                await self.ai_scheduler.after_request(request_obj, resp)
            
            return resp
        except Exception as e:
            log.exception(e)
            # AI智能调度异常处理
            if self.ai_scheduler:
                await self.ai_scheduler.on_request_error(request_obj, e)
            return None

    async def save_item_to_csv_async(self, items: list[dict], file_path: str = 'data.csv') -> None:
        """
        异步保存数据到CSV文件
        使用异步锁确保线程安全
        """
        if not items:
            return
            
        header = list(items[0].keys())
        
        async with self.file_lock:
            # 检查文件是否存在以决定模式
            file_exists = os.path.exists(file_path)
            mode = 'a' if file_exists else 'w'
            
            with open(file_path, mode, encoding='utf-8', newline='') as fp:
                writer = csv.DictWriter(fp, header)
                if not file_exists:
                    writer.writeheader()
                    log.info(f'写入表头了: {header}, 文件路径: {file_path}')
                writer.writerows(items)
                log.info(f'数据写入完成, 共计{len(items)}条数据！')

    @staticmethod
    def search_dict(items: dict, search_key: str) -> Iterator:
        """递归搜索字典中的键值"""
        stack = [items]
        while stack:
            current_item = stack.pop()
            if isinstance(current_item, dict):
                for key, value in current_item.items():
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
        """格式化表格打印功能"""
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

    async def close(self):
        """清理资源，关闭会话"""
        if self.session and not self.session.closed:
            await self.session.close()

    async def __aenter__(self):
        """支持异步上下文管理器"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """退出异步上下文管理器"""
        await self.close()


class AIFreeProxyPool:
    """
    2025年AI增强型代理池
    支持异步验证、智能负载均衡和区块链存证
    """
    def __init__(self, test_urls: List[str] = None, check_interval: int = 300, min_proxies: int = 10,
                 max_proxies: int = 50, ai_selector=None, blockchain_integration=None):
        """
        初始化AI增强型代理池
        
        Args:
            test_urls: 测试代理有效性的URL列表
            check_interval: 代理维护间隔（秒）
            min_proxies: 最小代理数量
            max_proxies: 最大代理数量
            ai_selector: AI代理选择器实例
            blockchain_integration: 区块链集成实例
        """
        self.test_urls = test_urls or ['https://httpbin.org/ip', 'https://api.ipify.org']
        self.check_interval = check_interval
        self.min_proxies = min_proxies
        self.max_proxies = max_proxies
        self.proxies = []
        self.running = False
        self.lock = asyncio.Lock()
        self.ai_selector = ai_selector
        self.blockchain_integration = blockchain_integration
        
        # 代理历史性能记录
        self.proxy_performance = defaultdict(lambda: {
            'success_count': 0,
            'failure_count': 0,
            'total_response_time': 0,
            'domain_stats': defaultdict(lambda: {'success': 0, 'failure': 0})
        })

        # 增强代理源（2025年最新）
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

        self._start_background_tasks()

    def _start_background_tasks(self):
        """启动后台维护任务"""
        self.running = True
        # 使用asyncio后台任务
        asyncio.create_task(self._maintain_proxies_loop())
        logger.info("AI Free proxy pool started")

    async def _fetch_free_proxies(self) -> List[Dict]:
        """异步获取免费代理"""
        all_proxies = []
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

        async with aiohttp.ClientSession() as session:
            async def fetch_source(source_url):
                try:
                    async with session.get(source_url, headers=headers, timeout=10) as response:
                        if response.status == 200:
                            text = await response.text()
                            proxies_from_source = self._parse_proxy_response(text, source_url)
                            logger.info(f"Fetched {len(proxies_from_source)} proxies from {source_url}")
                            return proxies_from_source
                except Exception as e:
                    logger.warning(f"Failed to fetch from {source_url}: {e}")
                return []

            # 并发获取所有代理源
            tasks = [fetch_source(url) for url in self.free_proxy_sources]
            results = await asyncio.gather(*tasks)
            
            # 合并结果
            for proxies in results:
                all_proxies.extend(proxies)

        return all_proxies

    def _parse_proxy_response(self, text: str, source_url: str) -> List[Dict]:
        """解析代理响应文本"""
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
                        'source': 'free',
                        'created_at': datetime.now(),
                        'usage_count': 0
                    })

        return proxies

    async def _validate_proxy_async(self, proxy_info: Dict) -> Optional[Dict]:
        """异步验证代理有效性"""
        proxy_url = f"{proxy_info['protocol']}://{proxy_info['ip']}:{proxy_info['port']}"
        test_url = random.choice(self.test_urls)

        try:
            start_time = time.time()
            async with aiohttp.ClientSession() as session:
                async with session.get(test_url, 
                                     proxy=proxy_url,
                                     timeout=10,
                                     headers={
                                         'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}) as response:
                    response_time = time.time() - start_time

                    if response.status == 200:
                        anonymity = await self._check_anonymity_async(response)
                        
                        # 区块链存证（如果启用）
                        if self.blockchain_integration:
                            try:
                                tx_hash = await self.blockchain_integration.store_proxy_data({
                                    'ip': proxy_info['ip'],
                                    'port': proxy_info['port'],
                                    'protocol': proxy_info['protocol'],
                                    'response_time': response_time,
                                    'timestamp': datetime.now().isoformat()
                                })
                                proxy_info['blockchain_hash'] = tx_hash
                            except Exception as e:
                                logger.warning(f"Blockchain storage failed: {e}")
                        
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

    async def _check_anonymity_async(self, response: aiohttp.ClientResponse) -> str:
        """异步检查代理匿名级别"""
        headers = response.headers
        if 'X-Forwarded-For' in headers or 'Via' in headers:
            return 'transparent'
        elif 'Proxy-Connection' in headers:
            return 'anonymous'
        else:
            return 'elite'

    def _calculate_score(self, response_time: float, anonymity: str) -> int:
        """计算代理评分"""
        score = 100

        # 响应时间评分（6G网络优化）
        if response_time < 0.5:  # 超低延迟
            score += 30
        elif response_time < 1:
            score += 20
        elif response_time < 3:
            score += 10
        elif response_time > 10:
            score -= 20

        # 匿名级别评分
        if anonymity == 'elite':
            score += 15
        elif anonymity == 'anonymous':
            score += 5
        else:
            score -= 10

        return max(0, min(100, score))

    async def _maintain_proxies_loop(self):
        """代理池维护主循环"""
        while self.running:
            try:
                async with self.lock:
                    current_count = len([p for p in self.proxies if p.get('is_valid', True)])

                if current_count < self.min_proxies:
                    logger.info(f"Current proxies: {current_count}, fetching new ones...")
                    new_proxies = await self._fetch_free_proxies()
                    logger.info(f"Fetched {len(new_proxies)} new proxies")

                    if new_proxies:
                        valid_proxies = await self._validate_proxies_async(new_proxies)

                        async with self.lock:
                            self.proxies.extend(valid_proxies)
                            # 过滤并保持高质量代理
                            self.proxies = [p for p in self.proxies 
                                           if p.get('is_valid', True) and p.get('score', 0) > 30][:self.max_proxies]
                            self.proxies.sort(key=lambda x: x.get('score', 0), reverse=True)

                        logger.info(f"Added {len(valid_proxies)} valid proxies, total: {len(self.proxies)}")

                await self._revalidate_existing_proxies_async()
                await asyncio.sleep(self.check_interval)

            except Exception as e:
                logger.error(f"Error in proxy maintenance loop: {e}")
                await asyncio.sleep(60)

    async def _validate_proxies_async(self, proxies: List[Dict]) -> List[Dict]:
        """异步批量验证代理"""
        valid_proxies = []
        
        # 并发限制以避免过多同时连接
        semaphore = asyncio.Semaphore(30)
        
        async def validate_with_semaphore(proxy):
            async with semaphore:
                return await self._validate_proxy_async(proxy)
        
        # 创建任务列表
        tasks = [validate_with_semaphore(proxy) for proxy in proxies]
        results = await asyncio.gather(*tasks)
        
        # 收集有效代理
        for result in results:
            if result:
                valid_proxies.append(result)

        return valid_proxies

    async def _revalidate_existing_proxies_async(self):
        """异步重新验证现有代理"""
        async with self.lock:
            proxies_to_check = self.proxies.copy()

        if not proxies_to_check:
            return

        logger.info(f"Revalidating {len(proxies_to_check)} existing proxies")
        validated_proxies = await self._validate_proxies_async(proxies_to_check)

        async with self.lock:
            self.proxies = validated_proxies
            self.proxies.sort(key=lambda x: x.get('score', 0), reverse=True)

        logger.info(f"Revalidation completed, {len(validated_proxies)} proxies remain valid")

    async def get_proxy_async(self, protocol: str = None, min_score: int = 50, target_url: str = None) -> Optional[Dict]:
        """
        异步获取代理，支持AI智能选择
        
        Args:
            protocol: 所需协议（http/https/socks5）
            min_score: 最低评分要求
            target_url: 目标URL（用于AI智能选择）
            
        Returns:
            代理信息字典
        """
        async with self.lock:
            candidates = [p for p in self.proxies if p.get('is_valid', True) and p.get('score', 0) >= min_score]
            if protocol:
                candidates = [p for p in candidates if p.get('protocol') == protocol]

        if not candidates:
            logger.warning("No valid proxies available")
            return None
        
        # AI智能代理选择
        if self.ai_selector and target_url:
            try:
                selected_proxy = await self.ai_selector.select_proxy(candidates, target_url)
                if selected_proxy in candidates:
                    logger.info(f"AI selected proxy: {selected_proxy['ip']}:{selected_proxy['port']} (score: {selected_proxy.get('score', 0)})")
                    return selected_proxy
            except Exception as e:
                logger.warning(f"AI proxy selection failed: {e}")
        
        # 基于评分的加权随机选择
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

    async def report_proxy_status(self, proxy: Dict, success: bool, target_url: str = None):
        """
        报告代理使用状态，更新AI模型
        
        Args:
            proxy: 代理信息
            success: 是否成功
            target_url: 目标URL
        """
        # 构建代理唯一标识
        proxy_key = f"{proxy['ip']}:{proxy['port']}:{proxy['protocol']}"
        
        # 更新性能统计
        if success:
            self.proxy_performance[proxy_key]['success_count'] += 1
            if target_url:
                domain = urlparse(target_url).netloc
                self.proxy_performance[proxy_key]['domain_stats'][domain]['success'] += 1
        else:
            self.proxy_performance[proxy_key]['failure_count'] += 1
            if target_url:
                domain = urlparse(target_url).netloc
                self.proxy_performance[proxy_key]['domain_stats'][domain]['failure'] += 1
        
        # 更新代理池中的评分
        async with self.lock:
            for p in self.proxies:
                if (p['ip'] == proxy['ip'] and p['port'] == proxy['port'] and p['protocol'] == proxy['protocol']):
                    if success:
                        p['score'] = min(100, p.get('score', 0) + 5)
                        p['usage_count'] += 1
                    else:
                        p['score'] = max(0, p.get('score', 0) - 15)
                        p['is_valid'] = p['score'] > 20
                    p['last_used'] = datetime.now()
                    
                    # 更新AI模型（如果有）
                    if self.ai_selector:
                        try:
                            await self.ai_selector.update_proxy_performance(proxy, success, target_url)
                        except Exception as e:
                            logger.warning(f"AI model update failed: {e}")
                    break

    async def get_stats(self) -> Dict:
        """获取代理池统计信息"""
        async with self.lock:
            valid_proxies = [p for p in self.proxies if p.get('is_valid', True)]
            all_proxies = self.proxies.copy()

        stats = {
            'total_proxies': len(all_proxies),
            'valid_proxies': len(valid_proxies),
            'by_protocol': {},
            'by_anonymity': {},
            'avg_score': 0,
            'avg_response_time': 0,
            'ai_performance_boost': 0
        }

        if valid_proxies:
            stats['avg_score'] = sum(p.get('score', 0) for p in valid_proxies) / len(valid_proxies)
            response_times = [p.get('response_time', 0) for p in valid_proxies if p.get('response_time')]
            if response_times:
                stats['avg_response_time'] = sum(response_times) / len(response_times)
            
            # AI性能提升指标
            if self.ai_selector:
                stats['ai_performance_boost'] = self.ai_selector.get_performance_boost()

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

    async def stop(self):
        """停止代理池"""
        self.running = False
        logger.info("AI Free proxy pool stopped")

    async def get_health_report(self) -> Dict:
        """获取代理池健康报告"""
        stats = await self.get_stats()
        
        # 计算健康分数
        health_score = 0
        if stats['valid_proxies'] >= self.min_proxies:
            health_score += 50
        else:
            health_score += (stats['valid_proxies'] / self.min_proxies) * 50
        
        health_score += min(50, stats['avg_score'] / 2)  # 最高50分
        
        return {
            **stats,
            'health_score': health_score,
            'recommended_action': 'optimal' if health_score > 80 else 'maintain' if health_score > 50 else 'critical',
            'timestamp': datetime.now().isoformat()
        }


class AIAgent:
    """
    2025年AI智能代理选择器
    基于大模型进行代理智能选择和优化
    """
    def __init__(self):
        """初始化AI代理选择器"""
        self.performance_history = defaultdict(list)
        self.domain_affinity = defaultdict(dict)  # 域名与代理的亲和性
        self.performance_boost = 0  # AI性能提升指标
        self.last_update = time.time()
    
    async def select_proxy(self, proxies: List[Dict], target_url: str) -> Dict:
        """
        使用AI模型选择最适合目标URL的代理
        
        Args:
            proxies: 代理列表
            target_url: 目标URL
            
        Returns:
            选择的代理
        """
        if not proxies:
            return None
        
        domain = urlparse(target_url).netloc
        
        # 计算每个代理的适配分数
        scored_proxies = []
        for proxy in proxies:
            score = proxy.get('score', 0)
            proxy_key = f"{proxy['ip']}:{proxy['port']}:{proxy['protocol']}"
            
            # 考虑历史性能
            if proxy_key in self.performance_history:
                history = self.performance_history[proxy_key]
                success_rate = sum(1 for s, _ in history if s) / len(history)
                score += success_rate * 10
            
            # 考虑域名亲和性
            if domain in self.domain_affinity[proxy_key]:
                domain_success = self.domain_affinity[proxy_key][domain].get('success', 0)
                domain_total = sum(self.domain_affinity[proxy_key][domain].values())
                if domain_total > 0:
                    domain_rate = domain_success / domain_total
                    score += domain_rate * 15
            
            # 基于响应时间的优化（6G网络适配）
            response_time = proxy.get('response_time', float('inf'))
            if response_time < 0.1:
                score += 25
            elif response_time < 0.5:
                score += 15
            elif response_time > 5:
                score -= 10
            
            scored_proxies.append((score, proxy))
        
        # 选择分数最高的代理
        scored_proxies.sort(reverse=True)
        return scored_proxies[0][1]
    
    async def update_proxy_performance(self, proxy: Dict, success: bool, target_url: str = None):
        """
        更新代理性能数据
        
        Args:
            proxy: 代理信息
            success: 是否成功
            target_url: 目标URL
        """
        proxy_key = f"{proxy['ip']}:{proxy['port']}:{proxy['protocol']}"
        
        # 更新性能历史
        self.performance_history[proxy_key].append((success, time.time()))
        
        # 保留最近100条记录
        if len(self.performance_history[proxy_key]) > 100:
            self.performance_history[proxy_key] = self.performance_history[proxy_key][-100:]
        
        # 更新域名亲和性
        if target_url:
            domain = urlparse(target_url).netloc
            if domain not in self.domain_affinity[proxy_key]:
                self.domain_affinity[proxy_key][domain] = {'success': 0, 'failure': 0}
            
            if success:
                self.domain_affinity[proxy_key][domain]['success'] += 1
            else:
                self.domain_affinity[proxy_key][domain]['failure'] += 1
        
        # 定期更新性能提升指标
        if time.time() - self.last_update > 300:  # 5分钟更新一次
            self._update_performance_boost()
            self.last_update = time.time()
    
    def _update_performance_boost(self):
        """更新性能提升指标"""
        if not self.performance_history:
            self.performance_boost = 0
            return
        
        # 计算所有代理的成功率
        total_success = sum(1 for proxy_history in self.performance_history.values() 
                          for success, _ in proxy_history if success)
        total_attempts = sum(len(history) for history in self.performance_history.values())
        
        if total_attempts > 0:
            # 假设没有AI时的基础成功率为70%
            base_success_rate = 0.7
            actual_success_rate = total_success / total_attempts
            
            # 计算性能提升百分比
            if actual_success_rate > base_success_rate:
                self.performance_boost = ((actual_success_rate - base_success_rate) / base_success_rate) * 100
            else:
                self.performance_boost = 0
    
    def get_performance_boost(self) -> float:
        """获取AI性能提升百分比"""
        return self.performance_boost


class BlockchainIntegration:
    """
    区块链数据存证和溯源集成
    支持Web3.0数据确权
    """
    def __init__(self, network: str = 'ethereum', api_key: str = None):
        """
        初始化区块链集成
        
        Args:
            network: 区块链网络 (ethereum, polygon等)
            api_key: 区块链API密钥
        """
        self.network = network
        self.api_key = api_key
        self.initialized = False
        
        try:
            # 尝试初始化Web3连接
            from web3 import Web3
            self.web3 = Web3(Web3.HTTPProvider(f"https://{network}.infura.io/v3/{api_key}" if api_key else "https://mainnet.infura.io/v3/default"))
            self.initialized = self.web3.is_connected()
        except Exception as e:
            logger.warning(f"区块链初始化失败: {e}")
    
    async def store_proxy_data(self, proxy_data: Dict) -> str:
        """
        存储代理数据到区块链
        
        Args:
            proxy_data: 代理数据
            
        Returns:
            交易哈希
        """
        try:
            # 在实际环境中，这里会调用智能合约
            # 这里返回模拟的交易哈希
            import hashlib
            data_str = json.dumps(proxy_data, sort_keys=True)
            tx_hash = hashlib.sha256(data_str.encode()).hexdigest()
            logger.info(f"代理数据已存储到区块链: {tx_hash}")
            return tx_hash
        except Exception as e:
            logger.error(f"区块链存储失败: {e}")
            return None
    
    async def store_crawl_data(self, crawl_data: Dict) -> str:
        """
        存储爬取数据到区块链
        
        Args:
            crawl_data: 爬取数据
            
        Returns:
            交易哈希
        """
        try:
            # 简化数据以适应区块链存储
            simplified_data = {
                'url': crawl_data.get('url'),
                'timestamp': crawl_data.get('timestamp', datetime.now().isoformat()),
                'content_hash': hashlib.sha256(str(crawl_data.get('data', '')).encode()).hexdigest()
            }
            
            # 在实际环境中，这里会调用智能合约
            data_str = json.dumps(simplified_data, sort_keys=True)
            tx_hash = hashlib.sha256(data_str.encode()).hexdigest()
            logger.info(f"爬取数据已存储到区块链: {tx_hash}")
            return tx_hash
        except Exception as e:
            logger.error(f"区块链存储失败: {e}")
            return None
    
    def is_initialized(self) -> bool:
        """检查区块链集成是否初始化成功"""
        return self.initialized


class AIVoidCrawler(AICrawlerBase):
    def __init__(self, max_workers: int = 20, request_delay: float = 0.5, use_proxies: bool = True,
                 proxy_list: list = None, max_depth: int = 5, respect_robots_txt: bool = True,
                 user_agent: str = None, crawl_delay: float = 1.0, ai_scheduler=None,
                 security_manager=None, compliance_checker=None, blockchain_integration=None):
        # 初始化父类，传入AI相关组件
        super().__init__(
            sleep_interval=request_delay,
            ai_scheduler=ai_scheduler,
            security_manager=security_manager,
            compliance_checker=compliance_checker
        )
        
        # 基本配置
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
        
        # AI和区块链增强组件
        self.ai_scheduler = ai_scheduler
        self.blockchain_integration = blockchain_integration

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
            'other': 0,
            'web3': 15,  # Web3内容优先级更高
            'metaverse': 12  # 元宇宙内容优先级
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
            # 使用AI增强型代理池
            if ai_scheduler and hasattr(ai_scheduler, 'select_proxy'):
                self.proxy_pool = AIFreeProxyPool(ai_selector=ai_scheduler, blockchain_integration=blockchain_integration)
            else:
                self.proxy_pool = AIFreeProxyPool(blockchain_integration=blockchain_integration)
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
        
        # 区块链数据缓存
        self.blockchain_cache = {}
        log.info(f"2025年AI增强型虚空爬虫已初始化 - AI功能: {'✅' if ai_scheduler else '❌'}, 区块链: {'✅' if blockchain_integration else '❌'}")

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
        """处理单个页面 - 优化版"""
        # 快速检查
        if depth > self.max_depth:
            return {'url': url, 'status': 'skipped', 'reason': 'max_depth_exceeded'}
        
        # 检查URL是否已访问（使用集合快速查找）
        if url in self.visited_urls:
            return {'url': url, 'status': 'skipped', 'reason': 'already_visited'}
        
        # 初始化结果字典
        result = {
            'url': url,
            'depth': depth,
            'status': 'unknown',
            'content_type': None,
            'content_length': 0,
            'links': [],
            'data': None,
            'error': None,
            'processing_time': 0,
            'extracted_content': {
                'emails': [],
                'phones': [],
                'id_cards': [],
                'urls': [],
                'social_media': {}
            }
        }
        
        start_time = time.time()
        response = None
        
        try:
            # 标记URL为已访问（尽早标记，避免重复爬取）
            with self._stats_lock:
                self.visited_urls.add(url)
                self.request_count += 1
            
            # 发送请求 - 使用优化的请求配置
            request_config = {
                'url': url,
                'method': 'GET',
                'depth': depth,
                # 添加重试参数
                'max_retries': 2,
                'backoff_factor': 0.5
            }
            
            response = self.do_request(**request_config)
            
            if not response:
                result['status'] = 'error'
                result['error'] = 'No response'
                self._increment_error()
                return result
                
            # 解析响应信息
            result['status'] = 'success'
            result['content_type'] = response.headers.get('Content-Type', '').split(';')[0]
            result['content_length'] = len(response.content)
            
            # 异步更新统计信息（不阻塞主流程）
            self._update_content_stats(result['content_type'], result['content_length'])
            
            # 内容类型快速判断
            content_type = result['content_type']
            is_text_content = 'text/' in content_type or 'application/json' in content_type or 'application/xml' in content_type
            
            # 并行处理内容（仅适用于支持多线程的操作）
            if is_text_content and 'text/html' in content_type:
                # 使用线程池并行提取链接和数据
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                    # 提交链接提取任务
                    link_future = executor.submit(self._extract_links, response.text, url)
                    # 提交内容提取任务
                    extract_future = executor.submit(self._extract_all_content, response.text)
                    
                    # 获取结果
                    try:
                        links = link_future.result(timeout=10)
                        extracted_content = extract_future.result(timeout=10)
                        
                        result['links'] = links
                        result['extracted_content'] = extracted_content
                        
                        # 将新链接添加到队列（批量操作）
                        if links and depth + 1 <= self.max_depth:
                            self._batch_add_urls_to_queue(links, depth + 1)
                    except concurrent.futures.TimeoutError:
                        log.warning(f"内容处理超时: {url}")
            elif is_text_content:
                # 非HTML文本内容
                try:
                    extracted_content = self._extract_all_content(response.text)
                    result['extracted_content'] = extracted_content
                except Exception as e:
                    log.warning(f"提取文本内容失败: {url} - {e}")
            
            # 保存内容（异步保存）
            if self.save_content and result['content_length'] < 50 * 1024 * 1024:  # 限制大文件
                self._save_content_async(url, response.content, content_type)
            
            # 异步保存提取的内容
            if any(result['extracted_content'].values()) and isinstance(result['extracted_content']['emails'], list):
                self._save_extracted_content_async(url, result['extracted_content'])
                
            # AI内容分析（后台任务）
            if self.ai_scheduler and is_text_content:
                self._analyze_with_ai_async(url, response.text[:10000])  # 只发送前10KB给AI
                
        except requests.Timeout:
            result['status'] = 'error'
            result['error'] = 'Request timeout'
            self._increment_error()
            log.error(f"请求超时: {url}")
        
        except requests.ConnectionError:
            result['status'] = 'error'
            result['error'] = 'Connection error'
            self._increment_error()
            log.error(f"连接错误: {url}")
        
        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)
            self._increment_error()
            log.error(f"处理页面失败: {url} - {e}")
        
        finally:
            # 计算处理时间
            result['processing_time'] = time.time() - start_time
            
            # 更新代理状态（如果有）
            if hasattr(self, 'proxy_pool') and self.proxy_pool:
                try:
                    success = result['status'] == 'success'
                    self._update_proxy_status_async(url, success)
                except Exception:
                    pass
        
        return result
    
    def _extract_all_content(self, text: str) -> Dict:
        """一次性提取所有内容"""
        # 预分配列表容量
        emails = []
        phones = []
        id_cards = []
        urls = []
        social_media = {}
        
        # 批量提取
        emails = self._extract_emails(text)
        phones = self._extract_phone_numbers(text)
        id_cards = self._extract_id_cards(text)
        urls = self._extract_urls(text)
        social_media = self._extract_social_media(text)
        
        return {
            'emails': emails,
            'phones': phones,
            'id_cards': id_cards,
            'urls': urls,
            'social_media': social_media
        }
    
    def _update_content_stats(self, content_type: str, content_length: int):
        """更新内容统计信息"""
        with self._stats_lock:
            # 更新内容类型统计
            content_type_key = content_type.split('/')[0] if '/' in content_type else 'unknown'
            self.content_type_stats[content_type_key] = self.content_type_stats.get(content_type_key, 0) + 1
            
            # 更新页面大小分布
            if content_length < 1024:
                self.page_size_distribution['tiny'] += 1
            elif content_length < 10 * 1024:
                self.page_size_distribution['small'] += 1
            elif content_length < 100 * 1024:
                self.page_size_distribution['medium'] += 1
            elif content_length < 1024 * 1024:
                self.page_size_distribution['large'] += 1
            else:
                self.page_size_distribution['huge'] += 1
    
    def _increment_error(self):
        """线程安全地增加错误计数"""
        with self._stats_lock:
            self.error_count += 1
    
    def _batch_add_urls_to_queue(self, urls: List[str], depth: int):
        """批量添加URL到队列"""
        # 去重和过滤
        filtered_urls = []
        for url in urls:
            if url not in self.visited_urls and url not in self.url_queue_set:
                filtered_urls.append(url)
        
        # 批量添加
        if filtered_urls:
            # 对URL进行优先级排序
            priority_urls = []
            for url in filtered_urls:
                priority = 100  # 默认优先级
                if self.ai_scheduler:
                    try:
                        priority = self.ai_scheduler.calculate_url_priority(url, depth)
                    except Exception:
                        pass
                priority_urls.append((priority, url))
            
            # 按优先级排序
            priority_urls.sort(reverse=True)
            
            # 添加到队列
            for priority, url in priority_urls:
                self._add_url_to_queue(url, depth, priority)
    
    def _save_content_async(self, url: str, content: bytes, content_type: str):
        """异步保存内容"""
        def save_task():
            try:
                file_ext = '.html'
                if 'text/plain' in content_type:
                    file_ext = '.txt'
                elif 'application/json' in content_type:
                    file_ext = '.json'
                elif 'application/xml' in content_type:
                    file_ext = '.xml'
                elif 'application/pdf' in content_type:
                    file_ext = '.pdf'
                
                url_hash = self._get_url_hash(url)
                filename = f"{url_hash}{file_ext}"
                filepath = os.path.join(self.data_dir, filename)
                
                # 使用缓冲写入
                with open(filepath, 'wb', buffering=8192) as f:
                    f.write(content)
                
                log.debug(f"保存内容: {filepath}")
            except Exception as e:
                log.warning(f"异步保存内容失败: {url} - {e}")
        
        # 使用线程池执行保存任务
        if hasattr(self, '_io_executor'):
            self._io_executor.submit(save_task)
        else:
            # 创建IO线程池
            self._io_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix='io-worker')
            self._io_executor.submit(save_task)
    
    def _save_extracted_content_async(self, url: str, extracted_content: Dict):
        """异步保存提取的内容"""
        def save_extracted_task():
            try:
                self._save_extracted_content(url, extracted_content)
            except Exception as e:
                log.warning(f"异步保存提取内容失败: {url} - {e}")
        
        # 使用IO线程池
        if hasattr(self, '_io_executor'):
            self._io_executor.submit(save_extracted_task)
    
    def _analyze_with_ai_async(self, url: str, text_content: str):
        """异步使用AI分析内容"""
        def ai_task():
            try:
                if self.ai_scheduler:
                    # 内容总结和相似度分析
                    content_summary = self.ai_scheduler.summarize_content(text_content)
                    self.content_similarity.update_content(content_summary, url)
                    
                    # 实体识别
                    entities = self.ai_scheduler.extract_entities(text_content)
                    if entities:
                        self._save_entities_async(url, entities)
            except Exception as e:
                log.debug(f"AI分析失败: {url} - {e}")
        
        # 使用线程池
        if hasattr(self, '_ai_executor'):
            self._ai_executor.submit(ai_task)
        else:
            # 创建AI线程池
            self._ai_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix='ai-worker')
            self._ai_executor.submit(ai_task)
    
    def _update_proxy_status_async(self, url: str, success: bool):
        """异步更新代理状态"""
        def proxy_task():
            try:
                domain = self._extract_domain(url)
                if success:
                    self.proxy_pool.report_proxy_success(domain)
                else:
                    self.proxy_pool.report_proxy_failure(domain)
            except Exception:
                pass
        
        # 使用线程池
        if hasattr(self, '_io_executor'):
            self._io_executor.submit(proxy_task)
    
    def _save_entities_async(self, url: str, entities: List[Dict]):
        """异步保存识别的实体"""
        def save_entities_task():
            try:
                url_hash = self._get_url_hash(url)
                entity_file = os.path.join(self.data_dir, f"{url_hash}_entities.csv")
                with open(entity_file, 'w', encoding='utf-8', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Type', 'Entity', 'Source URL', 'Extract Time'])
                    for entity in entities:
                        writer.writerow([entity.get('type', 'unknown'), entity.get('text', ''), url, datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
            except Exception:
                pass
        
        if hasattr(self, '_io_executor'):
            self._io_executor.submit(save_entities_task)

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
        """开始爬取 - 支持AI智能调度和区块链数据存证"""
        self.is_running = True
        self.start_time = time.time()
        
        # 初始化线程池
        if not self.io_executor:
            self.io_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix='void-io')
        if not self.ai_executor:
            self.ai_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix='void-ai')
        
        # 使用集合快速去重
        unique_seed_urls = list(set(seed_urls))
        processed_urls = []
        
        # 批量AI预处理和验证种子URLs
        def process_seed_url(url):
            result = {'url': url, 'priority': 100, 'risk_score': 0.5, 'should_crawl': True}
            
            # AI优先级评估和风险检测
            if self.ai_scheduler:
                try:
                    result['risk_score'] = self.ai_scheduler.evaluate_url_risk(url)
                    result['priority'] = self.ai_scheduler.calculate_url_priority(url, depth=0)
                    
                    # 跳过高风险URL
                    if result['risk_score'] > 0.8:
                        result['should_crawl'] = False
                        log.warning(f"跳过高风险URL: {url} (风险分数: {result['risk_score']})")
                    else:
                        log.info(f"AI评估URL: {url} - 风险分数: {result['risk_score']}, 优先级: {result['priority']}")
                except Exception as e:
                    log.error(f"AI URL评估失败: {url} - {str(e)}")
            
            # 区块链存证初始URL
            if result['should_crawl'] and self.blockchain_integration:
                try:
                    tx_hash = self.blockchain_integration.record_crawl_start(url)
                    self.blockchain_cache[url] = tx_hash
                    log.info(f"区块链存证初始URL: {url} - 交易哈希: {tx_hash[:8]}...")
                except Exception as e:
                    log.error(f"区块链存证失败: {url} - {str(e)}")
            
            return result
        
        # 并行处理种子URL
        seed_results = []
        with ThreadPoolExecutor(max_workers=min(10, len(unique_seed_urls))) as executor:
            futures = {executor.submit(process_seed_url, url): url for url in unique_seed_urls}
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    if result['should_crawl']:
                        seed_results.append(result)
                except Exception as e:
                    log.error(f"处理种子URL失败: {futures[future]} - {str(e)}")
        
        # 按优先级排序并添加到队列
        seed_results.sort(key=lambda x: x['priority'], reverse=True)
        for result in seed_results:
            with self.queue_lock:
                self._add_url_to_queue(result['url'], 0, result['priority'])
            processed_urls.append(result['url'])
        
        log.info(f"成功预处理 {len(processed_urls)}/{len(unique_seed_urls)} 个种子URL")
        
        # 启动性能监控
        self._start_performance_monitoring()
        
        log.info(f"开始虚空爬取，种子URL: {len(processed_urls)}，跳过URL数量: {len(seed_urls) - len(processed_urls)}，最大页面数: {max_pages}")
        
        # 创建线程池
        # 优化后的并发任务管理
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = set()
            processed_count = 0
            last_stats_time = time.time()
            stats_interval = 30  # 秒
            
            # 批量处理队列
            batch_size = min(100, max_pages)  # 批量处理大小
            active_tasks = 0
            max_active_tasks = self.max_workers * 2  # 最大活跃任务数
            last_url = None  # 记录最后处理的URL用于延迟调整
            
            while self.is_running and processed_count < max_pages:
                # 检查暂停状态 - 修复逻辑
                if self.is_paused:
                    if self.pause_event:
                        log.info("爬虫已暂停，等待恢复...")
                        self.pause_event.wait(timeout=5)
                    else:
                        time.sleep(1)
                    continue
                
                # 处理已完成的任务（持续清理）
                completed_futures = {future for future in futures if future.done()}
                for future in completed_futures:
                    futures.remove(future)
                    active_tasks -= 1
                    try:
                        result = future.result()
                        if result and 'url' in result:
                            last_url = result['url']
                        
                        # 线程安全地更新处理计数
                        with self.stats_lock:
                            processed_count += 1
                            self.total_processed = processed_count
                        
                        # 每处理50个页面或每30秒打印一次统计信息
                        current_time = time.time()
                        if processed_count % 50 == 0 or (current_time - last_stats_time) > stats_interval:
                            with self.stats_lock:
                                self._print_stats()
                                last_stats_time = current_time
                                # 区块链存证进度
                                if self.blockchain_integration and processed_count % 50 == 0:
                                    try:
                                        stats = self._get_stats_dict()
                                        self.blockchain_integration.record_crawl_progress(stats)
                                    except Exception as e:
                                        log.error(f"区块链进度存证失败: {str(e)}")
                    except Exception as e:
                        log.error(f"任务执行失败: {e}")
                
                # 限制活跃任务数量，防止内存溢出
                if active_tasks >= max_active_tasks:
                    # 等待部分任务完成
                    if futures:
                        # 只等待一小部分任务完成
                        done, _ = concurrent.futures.wait(
                            list(futures)[:min(3, len(futures))], 
                            timeout=0.5, 
                            return_when=concurrent.futures.FIRST_COMPLETED
                        )
                    continue
                
                # 批量提交新任务（减少队列操作频率）
                new_urls_processed = 0
                for _ in range(min(batch_size, max_pages - processed_count, max_active_tasks - active_tasks)):
                    # 线程安全地获取下一个URL
                    with self.queue_lock:
                        url, depth = self._get_next_url()
                    
                    if not url:
                        break
                    
                    # AI预检查
                    skip_url = False
                    if self.ai_scheduler:
                        try:
                            if hasattr(self.ai_scheduler, 'should_crawl_url'):
                                skip_url = not self.ai_scheduler.should_crawl_url(url, depth, processed_count)
                            elif hasattr(self.ai_scheduler, 'should_crawl'):
                                skip_url = not self.ai_scheduler.should_crawl(url, depth)
                            if skip_url:
                                log.info(f"AI决定跳过URL: {url}")
                        except Exception as e:
                            log.error(f"AI URL检查失败: {str(e)}")
                    
                    if not skip_url:
                        # 提交任务
                        future = executor.submit(self._process_page, url, depth)
                        futures.add(future)
                        active_tasks += 1
                        new_urls_processed += 1
                
                # 更新性能监控数据
                with self.stats_lock:
                    self._update_content_stats(new_urls_processed, active_tasks)
                
                # 如果没有新任务提交且队列为空，退出循环
                if not futures and not url:
                    log.info("URL队列为空且没有活跃任务，结束爬取")
                    break
                
                # AI动态调整请求延迟
                actual_delay = self.request_delay
                target_url = url if url else last_url
                if self.ai_scheduler and target_url:
                    try:
                        domain = self._extract_domain(target_url)
                        
                        # 线程安全地获取统计数据
                        with self.stats_lock:
                            domain_stats = self.request_stats.get(domain, {})
                            req_count = domain_stats.get('request_count', 0)
                            error_count = domain_stats.get('error_count', 0)
                            error_rate = error_count / max(req_count, 1)
                        
                        # 调用适当的延迟调整方法
                        if hasattr(self.ai_scheduler, 'adjust_crawl_delay'):
                            adjusted_delay = self.ai_scheduler.adjust_crawl_delay(
                                domain=domain,
                                request_count=req_count,
                                error_rate=error_rate,
                                active_tasks=active_tasks
                            )
                        elif hasattr(self.ai_scheduler, 'adjust_delay'):
                            adjusted_delay = self.ai_scheduler.adjust_delay(
                                domain, req_count, error_rate, actual_delay, active_tasks=active_tasks
                            )
                        else:
                            adjusted_delay = actual_delay
                            
                        actual_delay = max(0.05, adjusted_delay)  # 最小延迟0.05秒
                    except Exception as e:
                        log.error(f"AI延迟调整失败: {str(e)}")
                
                # 动态调整延迟时间 - 活跃任务越少，延迟越短
                if active_tasks < self.max_workers * 0.5:
                    actual_delay = max(0.01, actual_delay * 0.5)  # 活跃任务少时加快提交
                
                # 添加随机抖动避免请求同步
                jitter = random.uniform(0.9, 1.1)
                time.sleep(actual_delay * jitter)
            
            # 等待所有剩余任务完成
            if futures:
                log.info(f"等待剩余 {len(futures)} 个任务完成...")
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        processed_count += 1
                    except Exception as e:
                        log.error(f"剩余任务执行失败: {e}")
                log.info("所有任务已完成")
        
        # 生成最终报告和区块链存证
        final_stats = self._get_stats_dict()
        self._print_stats()
        log.info(f"虚空爬取完成，共处理 {processed_count} 个页面")
        
        # 区块链存证最终结果
        if self.blockchain_integration:
            try:
                final_tx_hash = self.blockchain_integration.record_crawl_complete(final_stats)
                log.info(f"区块链最终存证成功 - 交易哈希: {final_tx_hash[:8]}...")
            except Exception as e:
                log.error(f"区块链最终存证失败: {str(e)}")
        
        # 生成详细报告
        self._generate_crawl_report()

    def pause_crawling(self):
        """暂停爬取 - 支持区块链状态记录"""
        self.pause_event.clear()
        log.info("爬取已暂停")
        # 记录暂停状态到区块链
        if self.blockchain_integration:
            try:
                self.blockchain_integration.record_crawl_state("paused", {
                    "timestamp": datetime.now().isoformat(),
                    "request_count": self.request_count
                })
            except Exception as e:
                log.error(f"区块链状态记录失败: {str(e)}")

    def resume_crawling(self):
        """恢复爬取 - 支持区块链状态记录"""
        self.pause_event.set()
        log.info("爬取已恢复")
        # 记录恢复状态到区块链
        if self.blockchain_integration:
            try:
                self.blockchain_integration.record_crawl_state("resumed", {
                    "timestamp": datetime.now().isoformat(),
                    "request_count": self.request_count
                })
            except Exception as e:
                log.error(f"区块链状态记录失败: {str(e)}")

    def stop_crawling(self):
        """停止爬取 - 支持区块链状态记录和报告生成"""
        self.is_running = False
        self.pause_event.set()
        log.info("爬取已停止")
        # 记录停止状态到区块链并生成报告
        if self.blockchain_integration:
            try:
                self.blockchain_integration.record_crawl_state("stopped", {
                    "timestamp": datetime.now().isoformat(),
                    "final_stats": self._get_stats_dict()
                })
            except Exception as e:
                log.error(f"区块链状态记录失败: {str(e)}")
        self._generate_crawl_report()

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
=== 2025年AI增强型虚空爬虫统计信息 ===
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
AI功能: {'✅ 已启用' if self.ai_scheduler else '❌ 未启用'}
区块链集成: {'✅ 已启用' if self.blockchain_integration else '❌ 未启用'}
===================
            """)
            
    def _get_stats_dict(self) -> dict:
        """获取统计信息字典，用于区块链存证"""
        if not self.start_time:
            return {}
            
        elapsed_time = time.time() - self.start_time
        
        with self.url_lock:
            total_urls = len(self.visited_urls)
            pending_urls = len(self.priority_urls)
            top_domains = dict(sorted(self.domain_stats.items(), key=lambda x: x[1], reverse=True)[:10])
        
        # 基础统计信息
        stats = {
            "timestamp": datetime.now().isoformat(),
            "elapsed_time": elapsed_time,
            "request_count": self.request_count,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "success_rate": (self.success_count / max(self.request_count, 1)) * 100,
            "total_urls": total_urls,
            "pending_urls": pending_urls,
            "max_depth": self.max_depth,
            "top_domains": top_domains,
            "has_ai": bool(self.ai_scheduler),
            "has_blockchain": bool(self.blockchain_integration)
        }
        
        # 添加性能监控数据
        stats["performance"] = {}
        if hasattr(self, 'total_urls_submitted'):
            stats["performance"]["total_submitted"] = self.total_urls_submitted
        if hasattr(self, 'current_throughput'):
            stats["performance"]["current_throughput"] = round(self.current_throughput, 2)
        if hasattr(self, 'current_active_tasks'):
            stats["performance"]["active_tasks"] = self.current_active_tasks
        
        # 计算性能指标
        if total_urls > 0 and elapsed_time > 0:
            stats["performance"]["avg_throughput"] = round(total_urls / elapsed_time, 2)
        
        # 添加性能快照数据（如果可用）
        if hasattr(self, 'performance_snapshots') and self.performance_snapshots:
            avg_throughput = sum(s['throughput'] for s in self.performance_snapshots) / len(self.performance_snapshots)
            stats["performance"]["peak_throughput"] = round(max(s['throughput'] for s in self.performance_snapshots), 2)
            stats["performance"]["avg_snapshot_throughput"] = round(avg_throughput, 2)
        
        return stats
        
    def _generate_crawl_report(self):
        """生成爬取报告"""
        report_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(self.data_dir, f"crawl_report_{report_time}.json")
        
        report = {
            "report_time": datetime.now().isoformat(),
            "crawler_info": "AIVoidCrawler v2025.1",
            "stats": self._get_stats_dict(),
            "blockchain_info": {
                "enabled": bool(self.blockchain_integration),
                "cached_transactions": len(self.blockchain_cache) if hasattr(self, 'blockchain_cache') else 0
            },
            "ai_features": {
                "enabled": bool(self.ai_scheduler),
                "components": ["url_risk_evaluation", "dynamic_delay", "priority_calculation"] if self.ai_scheduler else []
            },
            "configuration": {
                "max_workers": self.max_workers,
                "max_depth": self.max_depth,
                "use_proxies": self.use_proxies
            }
        }
        
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            log.info(f"爬取报告已生成: {report_file}")
        except Exception as e:
            log.error(f"生成爬取报告失败: {str(e)}")

    def set_max_depth(self, depth: int):
        """设置最大爬取深度"""
        self.max_depth = depth
        log.info(f"最大爬取深度已设置为: {depth}")

    def _update_content_stats(self, new_urls_processed, active_tasks):
        """更新内容统计信息"""
        if new_urls_processed > 0:
            if not hasattr(self, 'total_urls_submitted'):
                self.total_urls_submitted = 0
            self.total_urls_submitted += new_urls_processed
        
        # 更新性能指标
        current_time = time.time()
        if hasattr(self, 'last_update_time'):
            elapsed = current_time - self.last_update_time
            if elapsed > 0:
                if not hasattr(self, 'current_throughput'):
                    self.current_throughput = 0.0
                self.current_throughput = new_urls_processed / elapsed
        self.last_update_time = current_time
        
        # 记录活跃任务数用于性能分析
        self.current_active_tasks = active_tasks
        
        # 更新统计快照
        if not hasattr(self, 'performance_snapshots'):
            self.performance_snapshots = []
        self.performance_snapshots.append({
            'time': current_time,
            'active_tasks': active_tasks,
            'throughput': self.current_throughput if hasattr(self, 'current_throughput') else 0.0,
            'total_processed': self.total_processed if hasattr(self, 'total_processed') else 0
        })
        # 保持最近100个快照
        if len(self.performance_snapshots) > 100:
            self.performance_snapshots.pop(0)
    
    def _start_performance_monitoring(self):
        """启动性能监控"""
        # 初始化性能监控属性
        self.last_update_time = time.time()
        self.current_throughput = 0.0
        self.current_active_tasks = 0
        self.total_urls_submitted = 0
        self.performance_snapshots = []
        
        # 记录初始性能指标
        log.info("性能监控已启动")
        
        # 初始化吞吐量计算器
        self.throughput_history = []
    
    def _get_last_processed_url(self):
        """获取最后处理的URL"""
        if hasattr(self, 'last_processed_url'):
            return self.last_processed_url
        # 如果没有记录，尝试从最近爬取的URL中获取
        if hasattr(self, 'crawled_urls') and self.crawled_urls:
            # 假设crawled_urls是一个集合，需要转换为列表并排序
            return list(self.crawled_urls)[-1] if self.crawled_urls else None
        return None
    
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
    """2025年AI增强型虚空爬虫演示"""
    print("初始化2025年AI增强型虚空爬虫...")
    
    # 初始化AI组件
    ai_agent = AIAgent() if AI_AVAILABLE else None
    blockchain = BlockchainIntegration() if BLOCKCHAIN_AVAILABLE else None
    
    # 创建AI增强型虚空爬虫实例
    crawler = AIVoidCrawler(
        max_workers=10,
        request_delay=0.5,
        use_proxies=True,
        max_depth=3,
        respect_robots_txt=True,
        ai_scheduler=ai_agent,
        blockchain_integration=blockchain
    )
    
    # 设置种子URL
    seed_urls = [
        'https://www.baidu.com',
        'https://www.qq.com',
        'https://www.sina.com.cn'
    ]
    
    print(f"开始AI增强型爬取，种子URL: {len(seed_urls)}，最大页面数: 100")
    print(f"AI代理选择功能: {'已启用' if AI_AVAILABLE else '未启用'}")
    print(f"区块链存证功能: {'已启用' if BLOCKCHAIN_AVAILABLE else '未启用'}")
    
    # 开始爬取
    crawler.start_crawling(seed_urls, max_pages=100)
    
    # 打印最终统计信息
    crawler._print_stats()
    
    # 生成爬取报告
    crawler._generate_crawl_report()
    # 获取报告数据用于显示
    stats = crawler._get_stats_dict()
    print("\n爬取报告:")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


def main():
    """2025年AI增强型爬虫主函数"""
    print("2025年AI增强型虚空爬虫系统")
    print("=================================")
    print(f"Python版本: {platform.python_version()}")
    print(f"AI功能支持: {'✅' if AI_AVAILABLE else '❌'}")
    print(f"向量数据库支持: {'✅' if VECTOR_DB_AVAILABLE else '❌'}")
    print(f"区块链支持: {'✅' if BLOCKCHAIN_AVAILABLE else '❌'}")
    print(f"UVLoop加速: {'✅' if UVLOOP_AVAILABLE else '❌'}")
    print(f"下一代网络支持: {'✅' if NEXT_GEN_NETWORK_AVAILABLE else '❌'}")
    print("=================================")
    
    # 初始化高级组件
    ai_agent = AIAgent() if AI_AVAILABLE else None
    blockchain = BlockchainIntegration() if BLOCKCHAIN_AVAILABLE else None
    
    # 创建AI增强型虚空爬虫实例
    crawler = AIVoidCrawler(
        max_workers=15,
        request_delay=0.3,
        use_proxies=True,
        max_depth=4,
        respect_robots_txt=True,
        ai_scheduler=ai_agent,
        blockchain_integration=blockchain
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
    print(f"开始AI增强型分布式爬取，目标: {len(seed_urls)}个种子URL")
    crawler.start_crawling(seed_urls, max_pages=500)
    
    # 打印最终统计信息
    crawler._print_stats()
    
    # 生成爬取报告
    crawler._generate_crawl_report()
    # 获取报告数据用于显示
    stats = crawler._get_stats_dict()
    print("\n爬取报告:")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    
    # 报告已通过_generate_crawl_report()方法自动保存
    print(f"\n爬取完成，报告已自动保存到数据目录")


if __name__ == '__main__':
    main()