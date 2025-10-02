import threading
import hashlib
import os
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import asyncio
import aiohttp
from fake_useragent import UserAgent

# ... existing code ...

class PhoneIDCardCrawler(HighConcurrencyCrawler):
    """高并发手机号和身份证爬虫"""
    
    def __init__(self, max_workers: int = 20, request_delay: float = 0.05, use_proxies: bool = True):
        super().__init__(max_workers, request_delay, use_proxies)
        
        # 手机号正则表达式
        self.phone_patterns = [
            r'1[3-9]\d{9}',  # 标准手机号
            r'\+86\s*1[3-9]\d{9}',  # 带国际区号
            r'1[3-9]\d{1}\s*\d{4}\s*\d{4}',  # 带空格格式
        ]
        
        # 身份证正则表达式
        self.id_card_patterns = [
            r'[1-9]\d{5}(18|19|20)\d{2}((0[1-9])|(1[0-2]))(([0-2][1-9])|10|20|30|31)\d{3}[0-9Xx]',  # 18位
            r'[1-9]\d{5}\d{2}((0[1-9])|(1[0-2]))(([0-2][1-9])|10|20|30|31)\d{3}',  # 15位
        ]
        
        # 数据源配置
        self.data_sources = {
            'phone_sources': [
                'http://www.sjgsd.com/kf/gh.html',  # 手机号归属地查询
                'http://www.ip138.com/mobile.asp',  # 手机号查询
                'https://www.so.com/s?q=手机号',  # 搜索引擎
            ],
            'id_card_sources': [
                'http://www.ip138.com/idcard/',  # 身份证查询
                'https://www.so.com/s?q=身份证',  # 搜索引擎
                'http://id.8684.cn/',  # 身份证查询
            ]
        }
        
        # 存储结果
        self.phone_results = []
        self.id_card_results = []
        self.duplicate_check = set()
        
    def extract_phone_numbers(self, text: str) -> list:
        """从文本中提取手机号"""
        phones = []
        for pattern in self.phone_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                # 清理格式
                phone = re.sub(r'\s+', '', str(match))
                if len(phone) == 11 and phone not in self.duplicate_check:
                    phones.append(phone)
                    self.duplicate_check.add(phone)
        return phones
    
    def extract_id_cards(self, text: str) -> list:
        """从文本中提取身份证号"""
        id_cards = []
        for pattern in self.id_card_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                id_card = str(match)
                if id_card not in self.duplicate_check:
                    # 验证身份证有效性
                    if self.validate_id_card(id_card):
                        id_cards.append(id_card)
                        self.duplicate_check.add(id_card)
        return id_cards
    
    def validate_id_card(self, id_card: str) -> bool:
        """验证身份证号有效性"""
        if len(id_card) == 18:
            # 18位身份证验证
            factors = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]
            check_codes = ['1', '0', 'X', '9', '8', '7', '6', '5', '4', '3', '2']
            
            try:
                total = sum(int(id_card[i]) * factors[i] for i in range(17))
                remainder = total % 11
                return id_card[-1].upper() == check_codes[remainder]
            except:
                return False
        elif len(id_card) == 15:
            # 15位身份证简单验证
            return id_card.isdigit()
        return False
    
    def crawl_phone_numbers(self, search_terms: list = None, max_pages: int = 10) -> list:
        """爬取手机号"""
        if search_terms is None:
            search_terms = ['手机号', '电话号码', '联系方式']
            
        urls = []
        for term in search_terms:
            # 生成搜索引擎URL
            for page in range(1, max_pages + 1):
                urls.extend([
                    f'https://www.baidu.com/s?wd={term}&pn={(page-1)*10}',
                    f'https://www.so.com/s?q={term}&pn={page}',
                    f'https://www.sogou.com/web?query={term}&page={page}'
                ])
        
        log.info(f"开始爬取手机号，共{len(urls)}个URL")
        
        results = self.crawl_urls_concurrently(
            urls, 
            callback=self.process_phone_page
        )
        
        log.info(f"手机号爬取完成，找到{len(self.phone_results)}个手机号")
        return self.phone_results
    
    def crawl_id_cards(self, search_terms: list = None, max_pages: int = 10) -> list:
        """爬取身份证号"""
        if search_terms is None:
            search_terms = ['身份证', '身份证号', '身份信息']
            
        urls = []
        for term in search_terms:
            # 生成搜索引擎URL
            for page in range(1, max_pages + 1):
                urls.extend([
                    f'https://www.baidu.com/s?wd={term}&pn={(page-1)*10}',
                    f'https://www.so.com/s?q={term}&pn={page}',
                    f'https://www.sogou.com/web?query={term}&page={page}'
                ])
        
        log.info(f"开始爬取身份证号，共{len(urls)}个URL")
        
        results = self.crawl_urls_concurrently(
            urls, 
            callback=self.process_id_card_page
        )
        
        log.info(f"身份证号爬取完成，找到{len(self.id_card_results)}个身份证号")
        return self.id_card_results
    
    def process_phone_page(self, response) -> dict:
        """处理手机号页面"""
        try:
            soup = BeautifulSoup(response.text, 'html.parser')
            text = soup.get_text()
            
            # 提取手机号
            phones = self.extract_phone_numbers(text)
            
            # 提取页面中的其他信息
            page_info = {
                'title': soup.title.string if soup.title else '',
                'url': response.url,
                'phones_found': phones,
                'timestamp': datetime.now().isoformat()
            }
            
            # 保存结果
            for phone in phones:
                phone_data = {
                    'phone': phone,
                    'source_url': response.url,
                    'page_title': page_info['title'],
                    'crawl_time': page_info['timestamp'],
                    'type': 'phone'
                }
                self.phone_results.append(phone_data)
            
            return page_info
            
        except Exception as e:
            log.error(f"处理手机号页面错误: {e}")
            return {'error': str(e), 'url': response.url}
    
    def process_id_card_page(self, response) -> dict:
        """处理身份证页面"""
        try:
            soup = BeautifulSoup(response.text, 'html.parser')
            text = soup.get_text()
            
            # 提取身份证号
            id_cards = self.extract_id_cards(text)
            
            # 提取页面中的其他信息
            page_info = {
                'title': soup.title.string if soup.title else '',
                'url': response.url,
                'id_cards_found': id_cards,
                'timestamp': datetime.now().isoformat()
            }
            
            # 保存结果
            for id_card in id_cards:
                id_data = {
                    'id_card': id_card,
                    'source_url': response.url,
                    'page_title': page_info['title'],
                    'crawl_time': page_info['timestamp'],
                    'type': 'id_card'
                }
                self.id_card_results.append(id_data)
            
            return page_info
            
        except Exception as e:
            log.error(f"处理身份证页面错误: {e}")
            return {'error': str(e), 'url': response.url}
    
    def save_results(self, filename: str = None):
        """保存结果到文件"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'crawl_results_{timestamp}.csv'
        
        all_results = self.phone_results + self.id_card_results
        
        if all_results:
            self.save_item_to_csv(iter(all_results), filename)
            log.info(f"结果已保存到: {filename}")
        else:
            log.warning("没有找到任何结果")
    
    def get_statistics(self) -> dict:
        """获取统计信息"""
        return {
            'total_requests': self.request_count,
            'successful_requests': self.success_count,
            'failed_requests': self.error_count,
            'success_rate': self.success_count / self.request_count * 100 if self.request_count > 0 else 0,
            'phones_found': len(self.phone_results),
            'id_cards_found': len(self.id_card_results),
            'unique_entries': len(self.duplicate_check)
        }


class AdvancedPhoneIDCrawler(PhoneIDCardCrawler):
    """高级手机号和身份证爬虫，支持更多功能"""
    
    def __init__(self, max_workers: int = 30, request_delay: float = 0.03, use_proxies: bool = True):
        super().__init__(max_workers, request_delay, use_proxies)
        
        # 高级配置
        self.max_retries = 3
        self.timeout = 30
        self.respect_robots_txt = True
        
    async def async_crawl(self, urls: list, callback: callable) -> list:
        """异步爬取"""
        async with aiohttp.ClientSession() as session:
            tasks = []
            for url in urls:
                task = self._async_fetch(session, url, callback)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return [r for r in results if not isinstance(r, Exception)]
    
    async def _async_fetch(self, session, url: str, callback: callable) -> dict:
        """异步获取页面"""
        headers = self.get_random_headers()
        
        try:
            async with session.get(url, headers=headers, timeout=self.timeout) as response:
                text = await response.text()
                return await callback(text, url, response.status)
        except Exception as e:
            log.error(f"异步请求失败: {url} - {e}")
            return {'error': str(e), 'url': url}
    
    def crawl_with_keywords(self, keywords: list, search_engine: str = 'baidu', max_pages: int = 5) -> list:
        """根据关键词爬取"""
        urls = self.generate_search_urls(keywords, search_engine, max_pages)
        return self.crawl_urls_concurrently(urls)
    
    def generate_search_urls(self, keywords: list, search_engine: str, max_pages: int) -> list:
        """生成搜索URL"""
        urls = []
        base_urls = {
            'baidu': 'https://www.baidu.com/s?wd=',
            'google': 'https://www.google.com/search?q=',
            'bing': 'https://www.bing.com/search?q=',
            'sogou': 'https://www.sogou.com/web?query='
        }
        
        base_url = base_urls.get(search_engine, base_urls['baidu'])
        
        for keyword in keywords:
            for page in range(1, max_pages + 1):
                if search_engine == 'baidu':
                    url = f"{base_url}{keyword}&pn={(page-1)*10}"
                elif search_engine == 'google':
                    url = f"{base_url}{keyword}&start={(page-1)*10}"
                else:
                    url = f"{base_url}{keyword}&page={page}"
                
                urls.append(url)
        
        return urls


# 使用示例
def main():
    """主函数示例"""
    # 创建爬虫实例
    crawler = AdvancedPhoneIDCrawler(max_workers=25, use_proxies=True)
    
    # 爬取手机号
    print("开始爬取手机号...")
    phones = crawler.crawl_phone_numbers(['手机号', '联系方式'], max_pages=3)
    
    # 爬取身份证号
    print("开始爬取身份证号...")
    id_cards = crawler.crawl_id_cards(['身份证', '身份信息'], max_pages=3)
    
    # 显示统计信息
    stats = crawler.get_statistics()
    print("\n=== 爬取统计 ===")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # 保存结果
    crawler.save_results()
    
    # 显示部分结果
    if phones:
        print(f"\n找到的手机号示例: {phones[:3]}")
    if id_cards:
        print(f"找到的身份证号示例: {id_cards[:3]}")


if __name__ == "__main__":
    # 检查必要的库
    try:
        import bs4
        import fake_useragent
    except ImportError as e:
        print(f"缺少必要的库: {e}")
        print("请安装: pip install beautifulsoup4 fake-useragent aiohttp")
        sys.exit(1)
    
    main()