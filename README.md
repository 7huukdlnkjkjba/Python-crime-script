包括但不限于：
- 非法获取公民个人信息
- 网络攻击和入侵
- 数据窃取和贩卖
- 诈骗等违法犯罪活动

、
# 🚀 一个基于Python的高性能、高并发网络爬虫，专门用于提取网页中的手机号码和身份证信息。支持多线程、代理池、自动去重和数据验证。

## ✨ 特性

### 🎯 核心功能
- **高并发爬取** - 多线程并发请求，最高支持数百并发
- **智能数据提取** - 正则表达式匹配手机号和身份证
- **数据验证** - 身份证号码有效性校验
- **代理支持** - 内置免费代理池，自动维护和验证
- **多种数据源** - 支持URL列表、文件、搜索引擎等多种数据源

### 🔧 技术特性
- **请求重试机制** - 自动重试失败请求
- **随机User-Agent** - 防反爬虫检测
- **请求频率控制** - 可调节的请求延迟
- **详细统计信息** - 实时监控爬取状态
- **数据去重** - 自动过滤重复数据
- **多种格式导出** - CSV文件格式

## 🛠 安装依赖

```bash
pip install requests beautifulsoup4 fake-useragent aiohttp loguru prettytable retrying
```

## 📖 快速开始

### 基础使用

```python
from crawl import HighConcurrencyCrawler

# 创建爬虫实例
crawler = HighConcurrencyCrawler(
    max_workers=20,        # 并发线程数
    request_delay=0.1,     # 请求延迟(秒)
    use_proxies=False      # 是否使用代理
)

# 从URL列表爬取
urls = [
    'https://www.example.com/page1',
    'https://www.example.com/page2'
]

results = crawler.crawl_phone_and_id_from_urls(
    urls, 
    save_to_file='results.csv'
)

print(f"找到手机号: {len(results['phones'])} 个")
print(f"找到身份证: {len(results['id_cards'])} 个")
```

### 从文件爬取

```python
# 从文本文件提取URL并爬取
results = crawler.crawl_from_file(
    'urls.txt', 
    save_to_file='file_results.csv'
)
```

### 自定义爬取

```python
# 手动处理响应
def custom_callback(response):
    text = response.text
    # 自定义处理逻辑
    phones = crawler.extract_phone_numbers(text)
    id_cards = crawler.extract_id_cards(text)
    return {
        'phones': phones,
        'id_cards': id_cards,
        'content_length': len(text)
    }

results = crawler.crawl_urls_concurrently(
    urls, 
    callback=custom_callback
)
```

## 📊 输出格式

### 手机号数据
```csv
手机号,提取时间
13812345678,2024-01-01 12:00:00
13987654321,2024-01-01 12:00:01
```

### 身份证数据
```csv
id_card,area_code,birth_date,age,gender,is_valid,source_url,crawl_time
110101199001011234,110101,1990-01-01,34,男,是,https://example.com,2024-01-01 12:00:00
```

## ⚙️ 配置选项

### 爬虫参数
```python
crawler = HighConcurrencyCrawler(
    max_workers=20,        # 并发线程数 (1-100)
    request_delay=0.05,    # 请求间隔(秒)
    use_proxies=True,      # 启用代理池
    proxy_list=['http://proxy1:port', 'http://proxy2:port']  # 自定义代理
)
```

### 代理池配置
```python
from crawl import FreeProxyPool

proxy_pool = FreeProxyPool(
    test_urls=['https://httpbin.org/ip'],  # 代理测试URL
    check_interval=300,     # 代理检查间隔(秒)
    min_proxies=10,         # 最小代理数量
    max_proxies=50          # 最大代理数量
)
```

## 🔍 数据提取规则

### 手机号匹配
- 标准格式: `1[3-9]XXXXXXX`
- 带国际码: `+86 1[3-9]XXXXXXX`
- 带分隔符: `138-1234-5678`

### 身份证匹配
- 18位身份证号码
- 行政区划代码验证
- 出生日期验证
- 校验码验证

### 技术限制
- 🛡️ 支持HTTPS协议
- 🔄 自动处理编码问题
- ⏰ 内置请求频率限制
- ❌ 自动过滤无效数据

### 常见问题

1. **导入错误**
   ```bash
   # 安装缺失的库
   pip install beautifulsoup4 fake-useragent
   ```

2. **请求被阻塞**
   ```python
   # 增加延迟或使用代理
   crawler.set_request_delay(0.5)
   crawler.set_proxy_list(proxy_list)
   ```

3. **内存占用过高**
   ```python
   # 减少并发数
   crawler.set_max_workers(10)
   ```

### 日志查看
```python
from crawl import log
log.set_level('DEBUG')  # 设置日志级别
```

## 📁 项目结构

```
crawl.py                 # 主爬虫文件
├── CrawlBase           # 爬虫基类
├── HighConcurrencyCrawler # 高并发爬虫
├── FreeProxyPool       # 代理池管理
└── 工具函数            # 数据验证、导出等
```
## 工具正在意淫中...
