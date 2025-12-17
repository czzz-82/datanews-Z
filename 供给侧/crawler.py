import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import re
import time
import os
from urllib.parse import urljoin
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LoginSpider:
    def __init__(self, username, password):
        self.session = requests.Session()
        self.base_url = "https://report.zcmima.cn"
        self.login_url = f"{self.base_url}/index/user/login.html"
        self.username = username
        self.password = password
        self.is_logged_in = False
        
        # 设置请求头
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        self.session.headers.update(self.headers)
    
    def get_login_tokens(self):
        """获取登录页面所需的token"""
        try:
            response = self.session.get(self.login_url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 查找CSRF token或其他安全token
            token_input = soup.find('input', {'name': '__token__'})
            token = token_input['value'] if token_input else ''
            
            return {'token': token}
        except Exception as e:
            logger.error(f"获取登录token失败: {e}")
            return {}
    
    def login(self):
        """执行登录"""
        logger.info("正在登录...")
        
        try:
            # 首先获取token
            tokens = self.get_login_tokens()
            
            # 准备登录数据
            login_data = {
                'account': self.username,
                'password': self.password,
                'keeplogin': '1',
                '__token__': tokens.get('token', '')
            }
            
            # 发送登录请求
            response = self.session.post(
                self.login_url,
                data=login_data,
                headers={
                    'Content-Type': 'application/x-www-form-urlencoded',
                    'Referer': self.login_url
                },
                timeout=15,
                allow_redirects=True
            )
            
            # 检查登录是否成功
            if self.check_login_status():
                logger.info("✓ 登录成功")
                self.is_logged_in = True
                return True
            else:
                logger.error("✗ 登录失败，请检查用户名和密码")
                return False
                
        except Exception as e:
            logger.error(f"登录过程中出错: {e}")
            return False
    
    def check_login_status(self):
        """检查登录状态"""
        try:
            # 访问一个需要登录的页面来验证状态
            test_url = f"{self.base_url}/index/user/index.html"
            response = self.session.get(test_url, timeout=10)

            # 检查是否跳转到登录页或显示用户信息
            if "登录" in response.text and "password" in response.text:
                return False
            return True
            
        except Exception as e:
            logger.error(f"检查登录状态失败: {e}")
            return False
    
    def ensure_login(self):
        """确保处于登录状态"""
        if not self.is_logged_in:
            return self.login()
        return True

class EmploymentDataExtractor:
    def __init__(self, session=None):
        self.session = session or requests.Session()
        self.base_url = "https://report.zcmima.cn"
    
    def extract_school_info(self, html_content):
        """从HTML中提取学校信息和年份"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            title_tag = soup.find('title')
            if title_tag:
                title_text = title_tag.text
                # 匹配学校名称和年份，如"河北金融学院2022届毕业生就业质量报告"
                match = re.search(r'(.+?)(\d{4})届毕业生就业质量报告', title_text)
                if match:
                    school_name = match.group(1).strip()
                    year = match.group(2)
                    return school_name, year
            
            # 如果标题匹配失败，尝试从h1标签获取
            h1_tag = soup.find('h1', class_='tith1')
            if h1_tag:
                h1_text = h1_tag.text
                match = re.search(r'(.+?)(\d{4})届毕业生就业质量报告', h1_text)
                if match:
                    school_name = match.group(1).strip()
                    year = match.group(2)
                    return school_name, year
            
            return "未知学校", "未知年份"
        except Exception as e:
            logger.error(f"提取学校信息失败: {e}")
            return "未知学校", "未知年份"
    
    def extract_employment_rate_data(self, html_content):
        """从就业率页面提取专业数据"""
        try:
            school_name, year = self.extract_school_info(html_content)
            
            # 提取专业数据
            major_data = self.extract_major_data(html_content)
            
            data = {
                'school_name': school_name,
                'year': year,
                'data_type': 'employment_rate',
                'major_data': major_data
            }
            return data
        except Exception as e:
            logger.error(f"提取就业率数据失败: {e}")
            return None
    
    def extract_employment_flow_data(self, html_content):
        """从就业流向页面提取行业数据"""
        try:
            school_name, year = self.extract_school_info(html_content)
            
            # 提取行业数据
            industry_data = self.extract_industry_data(html_content)
            
            # 提取省份分布数据
            province_data = self.extract_province_data(html_content)
            
            data = {
                'school_name': school_name,
                'year': year,
                'data_type': 'employment_flow',
                'industry_data': industry_data,
                'province_data': province_data
            }
            return data
        except Exception as e:
            logger.error(f"提取就业流向数据失败: {e}")
            return None
    
    def extract_basic_stats_data(self, html_content):
        """从基本统计页面提取专业学生数据"""
        try:
            school_name, year = self.extract_school_info(html_content)
            
            # 提取专业学生数据（主要是student_number字段）
            major_data = self.extract_major_data(html_content)
            
            data = {
                'school_name': school_name,
                'year': year,
                'data_type': 'basic_stats',
                'major_data': major_data
            }
            return data
        except Exception as e:
            logger.error(f"提取基本统计数据失败: {e}")
            return None
    
    def extract_major_data(self, html_content):
        """提取专业数据"""
        try:
            # 从JavaScript变量中提取专业数据
            json_data = self.extract_json_data(html_content, 'data_json_str')
            if json_data:
                logger.info(f"extract_major_data: found JSON data, {len(json_data)} records")
                # 标准化字段并转换类型，保持与表格解析一致
                normalized = []
                for item in json_data:
                    try:
                        # 支持多种字段来源：优先使用明确字段，其次尝试 student_number 等备用字段
                        grad = None
                        if isinstance(item.get('graduate_number'), (int, float)):
                            grad = item.get('graduate_number')
                        elif isinstance(item.get('student_number'), (int, float)):
                            grad = item.get('student_number')
                        else:
                            grad = self.parse_number(str(item.get('graduate_number', '') or item.get('student_number', '')))

                        empn = None
                        if isinstance(item.get('employment_number'), (int, float)):
                            empn = item.get('employment_number')
                        else:
                            empn = self.parse_number(str(item.get('employment_number', '')))
                        # employment_rate 可能已经是数字或为字符串（如 '91' 或 '91%')
                        eraw = item.get('employment_rate', '')
                        er = None
                        if isinstance(eraw, (int, float)):
                            er = float(eraw)
                        else:
                            er = self.parse_percentage(str(eraw))

                        # 也保留 student_number 原始值（如果存在），以便后续判定
                        student_num = item.get('student_number')
                        normalized.append({
                            'education': item.get('education', ''),
                            'major': item.get('major', ''),
                            'graduate_number': int(grad) if grad is not None else 0,
                            'employment_number': int(empn) if empn is not None else 0,
                            'employment_rate': er,
                            'student_number': int(student_num) if isinstance(student_num, (int, float)) else (self.parse_number(str(student_num)) if student_num else None)
                        })
                    except Exception:
                        continue
                if not normalized:
                    logger.warning("extract_major_data: JSON extracted but normalization produced 0 records")
                    try:
                        debug_file = f"debug_major_{int(time.time())}.html"
                        with open(debug_file, 'w', encoding='utf-8') as f:
                            f.write(html_content)
                        logger.info(f"extract_major_data: saved debug HTML to {debug_file}")
                    except Exception as e:
                        logger.debug(f"无法保存 debug_major 文件: {e}")
                return normalized
            
            # 如果JavaScript提取失败，从表格中提取
            soup = BeautifulSoup(html_content, 'html.parser')
            table = soup.find('tbody', {'id': 'tb1'})
            if not table:
                logger.warning('extract_major_data: no tbody#tb1 found; returning empty list')
                try:
                    debug_file = f"debug_major_nobody_{int(time.time())}.html"
                    with open(debug_file, 'w', encoding='utf-8') as f:
                        f.write(html_content)
                    logger.info(f"extract_major_data: saved debug HTML to {debug_file}")
                except Exception as e:
                    logger.debug(f"无法保存 debug_major_nobody 文件: {e}")
                return []
            
            major_data = []
            rows = table.find_all('tr')
            for row in rows:
                cells = row.find_all('td')
                if len(cells) >= 5:
                    major_item = {
                        'education': cells[0].get_text(strip=True),
                        'major': cells[1].get_text(strip=True),
                        'graduate_number': self.parse_number(cells[2].get_text(strip=True)),
                        'employment_number': self.parse_number(cells[3].get_text(strip=True)),
                        'employment_rate': self.parse_percentage(cells[4].get_text(strip=True))
                    }
                    major_data.append(major_item)
            
            if not major_data:
                logger.warning('extract_major_data: parsed table but found 0 major rows')
                try:
                    debug_file = f"debug_major_rows_0_{int(time.time())}.html"
                    with open(debug_file, 'w', encoding='utf-8') as f:
                        f.write(html_content)
                    logger.info(f"extract_major_data: saved debug HTML to {debug_file}")
                except Exception as e:
                    logger.debug(f"无法保存 debug_major_rows_0 文件: {e}")
            return major_data
        except Exception as e:
            logger.error(f"提取专业数据失败: {e}")
            return []
    
    def extract_industry_data(self, html_content):
        """提取行业数据"""
        try:
            # 从JavaScript变量中提取行业数据
            json_data = self.extract_json_data(html_content, 'data_json_str')
            if json_data:
                logger.info(f"extract_industry_data: found JSON data, {len(json_data)} records")
                normalized = []
                for item in json_data:
                    try:
                        # 支持多种字段来源
                        fnum = None
                        if isinstance(item.get('flow_number'), (int, float)):
                            fnum = item.get('flow_number')
                        else:
                            fnum = self.parse_number(str(item.get('flow_number', '')))
                        
                        fraw = item.get('flow_rate', '')
                        fr = None
                        if isinstance(fraw, (int, float)):
                            fr = float(fraw)
                        else:
                            fr = self.parse_percentage(str(fraw))

                        normalized.append({
                            'education': item.get('education', ''),
                            'trade_name': item.get('trade_name', ''),
                            'flow_number': int(fnum) if fnum is not None else 0,
                            'flow_rate': fr
                        })
                    except Exception:
                        continue
                if not normalized:
                    logger.warning("extract_industry_data: JSON extracted but normalization produced 0 records")
                    try:
                        debug_file = f"debug_industry_{int(time.time())}.html"
                        with open(debug_file, 'w', encoding='utf-8') as f:
                            f.write(html_content)
                        logger.info(f"extract_industry_data: saved debug HTML to {debug_file}")
                    except Exception as e:
                        logger.debug(f"无法保存 debug_industry 文件: {e}")
                return normalized
            
            # 如果JavaScript提取失败，从表格中提取
            soup = BeautifulSoup(html_content, 'html.parser')
            table = soup.find('tbody', {'id': 'tb1'})
            if not table:
                logger.warning('extract_industry_data: no tbody#tb1 found; returning empty list')
                try:
                    debug_file = f"debug_industry_nobody_{int(time.time())}.html"
                    with open(debug_file, 'w', encoding='utf-8') as f:
                        f.write(html_content)
                    logger.info(f"extract_industry_data: saved debug HTML to {debug_file}")
                except Exception as e:
                    logger.debug(f"无法保存 debug_industry_nobody 文件: {e}")
                return []

            industry_data = []
            rows = table.find_all('tr')
            for row in rows:
                cells = row.find_all('td')
                if len(cells) >= 4:
                    industry_item = {
                        'education': cells[0].get_text(strip=True),
                        'trade_name': cells[1].get_text(strip=True),
                        'flow_number': self.parse_number(cells[2].get_text(strip=True)),
                        'flow_rate': self.parse_percentage(cells[3].get_text(strip=True))
                    }
                    industry_data.append(industry_item)

            if not industry_data:
                logger.warning('extract_industry_data: parsed table but found 0 industry rows')
                try:
                    debug_file = f"debug_industry_rows_0_{int(time.time())}.html"
                    with open(debug_file, 'w', encoding='utf-8') as f:
                        f.write(html_content)
                    logger.info(f"extract_industry_data: saved debug HTML to {debug_file}")
                except Exception as e:
                    logger.debug(f"无法保存 debug_industry_rows_0 文件: {e}")

            return industry_data
        except Exception as e:
            logger.error(f"提取行业数据失败: {e}")
            return []
    
    def extract_province_data(self, html_content):
        """提取省份分布数据"""
        try:
            # 优先使用通用的 JS 数组提取器尝试找出省份数据
            # 有些页面把数据放在 data: [..] 或 var xxx = [..] 中
            province = self.extract_json_data(html_content, var_name=None)
            if province:
                return province
            return []
        except Exception as e:
            logger.error(f"提取省份数据失败: {e}")
            return []

    def js_array_to_json(self, js_text):
        """把 JavaScript 风格的数组/对象文本尽量转换为标准 JSON 字符串后返回可解析的 JSON 对象
        处理的情况包括：单引号->双引号、去掉末尾逗号、把 undefined 替换为 null 等。"""
        try:
            s = js_text
            # 替换单引号包裹的字符串为双引号（尽量简单替换）
            s = re.sub(r"'([^']*)'", lambda m: '"' + m.group(1).replace('"', '\\"') + '"', s)
            # 把 undefined 转为 null
            s = re.sub(r"\bundefined\b", 'null', s)
            # 清理对象/数组中的末尾逗号
            s = re.sub(r",\s*([}\]])", r"\1", s)
            return s
        except Exception:
            return js_text
    
    def extract_json_data(self, html_content, var_name):
        """提取JSON格式的数据"""
        try:
            text = html_content

            # 如果调用者传入具体变量名，则优先按照 varName = [...] 或 var varName = [...] 等方式查找
            if var_name:
                # 支持多种写法：var/let/const name = [...]; name = [...]; name = '[...]';
                patterns = [
                    rf"(?:var|let|const)\s+{re.escape(var_name)}\s*=\s*(\[.*?\])",
                    rf"{re.escape(var_name)}\s*=\s*(\[.*?\])",
                    rf"(?:var|let|const)\s+{re.escape(var_name)}\s*=\s*['\"](\[.*?\])['\"]",
                    rf"{re.escape(var_name)}\s*=\s*['\"](\[.*?\])['\"]",
                ]
                for p in patterns:
                    m = re.search(p, text, re.DOTALL)
                    if m:
                        js_arr = m.group(1)
                        logger.debug(f"extract_json_data: matched pattern '{p}' for var_name={var_name}")
                        js_arr = self.js_array_to_json(js_arr)
                        return json.loads(js_arr)

            # 如果没有指定变量名或上面找不到，尝试查找 first occurrence of data: [ ... ] 或任意脚本里的首个数组文字
            # 1) 查找 data: [ ... ]
            m = re.search(r"data\s*:\s*(\[.*?\])", text, re.DOTALL)
            if m:
                js_arr = m.group(1)
                logger.debug("extract_json_data: matched data: [...] pattern")
                js_arr = self.js_array_to_json(js_arr)
                return json.loads(js_arr)

            # 2) 查找任何较大的数组字面量（谨慎匹配，避免匹配 HTML 属性或其它小数组）
            m2 = re.search(r"(\[\s*\{[\s\S]*?\}\s*\])", text, re.DOTALL)
            if m2:
                js_arr = m2.group(1)
                logger.debug("extract_json_data: matched generic array literal")
                js_arr = self.js_array_to_json(js_arr)
                return json.loads(js_arr)

            logger.debug(f"extract_json_data: no JSON array found for var_name={var_name}")
            return None
        except Exception as e:
            logger.error(f"提取JSON数据失败 {var_name}: {e}")
            
        return None
    
    def parse_number(self, text):
        """解析数字，移除'人'等后缀"""
        try:
            # 移除非数字字符，但保留小数点和负号
            number_text = re.sub(r'[^\d.-]', '', text)
            return int(number_text) if number_text else 0
        except:
            return 0
    
    def parse_percentage(self, text):
        """解析百分比"""
        try:
            # 移除百分号和非数字字符，但保留小数点
            percent_text = re.sub(r'[^\d.]', '', text)
            return float(percent_text) if percent_text else 0.0
        except:
            return 0.0

class AdvancedEmploymentCrawler:
    def __init__(self, username, password):
        self.login_spider = LoginSpider(username, password)
        self.data_extractor = EmploymentDataExtractor()
        self.all_data = []
    
    def login(self):
        """登录"""
        return self.login_spider.login()
    
    def get_school_list(self, max_pages=200):
        """获取学校列表"""
        if not self.login_spider.ensure_login():
            logger.error("无法登录，停止爬取")
            return []
            
        base_url = f"{self.login_spider.base_url}/report/school"
        all_schools = []
        
        for page in range(1, max_pages + 1):
            try:
                if page == 1:
                    url = base_url
                else:
                    url = f"{base_url}?page={page}"
                
                logger.info(f"正在获取第 {page} 页学校列表...")
                response = self.login_spider.session.get(url, timeout=15)
                
                if response.status_code != 200:
                    logger.error(f"第 {page} 页请求失败，状态码: {response.status_code}")
                    break
                
                # 检查是否被重定向到登录页
                if "登录" in response.text and "password" in response.text:
                    logger.warning("会话过期，重新登录...")
                    self.login_spider.is_logged_in = False
                    if not self.login_spider.ensure_login():
                        break
                    # 重新请求当前页
                    response = self.login_spider.session.get(url, timeout=15)
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # 解析学校卡片
                school_cards = soup.select('.college_item')
                logger.info(f"第 {page} 页找到 {len(school_cards)} 个学校")
                
                for card in school_cards:
                    try:
                        school_data = self.parse_school_card(card)
                        if school_data:
                            all_schools.append(school_data)
                    except Exception as e:
                        logger.error(f"解析学校卡片失败: {e}")
                        continue
                
                if not school_cards:
                    logger.info("没有找到学校卡片，可能已到最后一页")
                    break
                    
                time.sleep(1)  # 礼貌延时
                
            except Exception as e:
                logger.error(f"获取第 {page} 页时出错: {e}")
                break
        
        return all_schools
    
    def parse_school_card(self, card):
        """解析单个学校卡片"""
        try:
            # 学校名称
            name_elem = card.select_one('.college_name')
            school_name = name_elem.text.strip() if name_elem else "未知学校"
            
            # 学校信息
            info_elem = card.select_one('.college_info')
            if not info_elem:
                return None
            
            # 类型、层次、性质
            oth_elems = info_elem.select('.college_oth span')
            school_type = oth_elems[0].text if len(oth_elems) > 0 else ""
            education_level = oth_elems[1].text if len(oth_elems) > 1 else ""
            school_nature = oth_elems[2].text if len(oth_elems) > 2 else ""
            
            # 所在地
            location_elem = info_elem.select_one('.college_address')
            location = location_elem.text.replace('学校所在地：', '').strip() if location_elem else ""
            
            # 报告数量
            report_elems = card.select('.meet .color')
            ai_reports = int(report_elems[0].text) if len(report_elems) > 0 else 0
            data_reports = int(report_elems[1].text) if len(report_elems) > 1 else 0
            
            # 详情页URL
            onclick_attr = card.get('onclick', '')
            detail_url_match = re.search(r"window\.open\('([^']+)'\)", onclick_attr)
            detail_url = detail_url_match.group(1) if detail_url_match else ""
            
            # 学校ID
            school_id_match = re.search(r'school/(\d+)\.html', detail_url)
            school_id = school_id_match.group(1) if school_id_match else ""
            
            return {
                'school_id': school_id,
                'school_name': school_name,
                'school_type': school_type,
                'education_level': education_level,
                'school_nature': school_nature,
                'location': location,
                'ai_report_count': ai_reports,
                'data_report_count': data_reports,
                'detail_url': detail_url
            }
            
        except Exception as e:
            logger.error(f"解析学校卡片时出错: {e}")
            return None
    
    def get_school_data_screens(self, school_info):
        """获取学校的数据大屏链接"""
        if not self.login_spider.ensure_login():
            return None
            
        try:
            if not school_info.get('detail_url'):
                return None
                
            full_url = urljoin(self.login_spider.base_url, school_info['detail_url'])
            logger.info(f"正在获取学校详情: {school_info['school_name']}")
            
            response = self.login_spider.session.get(full_url, timeout=15)
            if response.status_code != 200:
                logger.error(f"学校详情请求失败: {response.status_code}")
                return None
            
            # 检查登录状态
            if "登录" in response.text and "password" in response.text:
                logger.warning("会话过期，重新登录...")
                self.login_spider.is_logged_in = False
                if not self.login_spider.ensure_login():
                    return None
                response = self.login_spider.session.get(full_url, timeout=15)
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 解析数据大屏报告
            data_reports = []
            data_items = soup.select('.tabs-panel:nth-of-type(2) .ml_item')
            for item in data_items:
                title = item.select_one('.com_name').text.strip()
                basic_url = item.select_one('.ml_btn')['href']
                
                # 从基本页面URL提取school_id和cid，生成就业率和流向的直接URL
                # 例如 /index/dataview/basic/sid/31331/cid/7.html 
                # 转为 /index/dataview/employmentrate/sid/31331/cid/7.html 和 employmentflow
                if 'basic' in basic_url:
                    # 替换 /basic/ 为 /employmentrate/ 和 /employmentflow/
                    employment_rate_url = basic_url.replace('/basic/', '/employmentrate/')
                    employment_flow_url = basic_url.replace('/basic/', '/employmentflow/')
                    
                    data_reports.append({
                        'title': title,
                        'employment_rate_url': employment_rate_url,
                        'employment_flow_url': employment_flow_url,
                        'basic_url': basic_url
                    })
            
            return data_reports
            
        except Exception as e:
            logger.error(f"获取学校数据大屏失败: {e}")
            return None
    
    def crawl_data_screen(self, url, school_name, report_title=None):
        """爬取单个数据大屏页面"""
        try:
            full_url = urljoin(self.login_spider.base_url, url)
            logger.info(f"正在获取数据: {full_url} (标题: {report_title})")
            
            response = self.login_spider.session.get(full_url, timeout=15)
            if response.status_code != 200:
                logger.error(f"数据请求失败: {response.status_code}")
                return None
            
            # 检查登录状态
            if "登录" in response.text and "password" in response.text:
                logger.warning("会话过期，重新登录...")
                self.login_spider.is_logged_in = False
                if not self.login_spider.ensure_login():
                    return None
                response = self.login_spider.session.get(full_url, timeout=15)
            
            # 根据URL和报告标题判断数据类型
            data = None
            if 'employmentflow' in url or '就业流向' in (report_title or ''):
                data = self.data_extractor.extract_employment_flow_data(response.text)
            elif 'employmentrate' in url or '就业率' in (report_title or ''):
                data = self.data_extractor.extract_employment_rate_data(response.text)
            elif '基本' in (report_title or '') or 'basic' in url:
                # 基本统计页面 - 只有学生人数，没有就业数据
                data = self.data_extractor.extract_basic_stats_data(response.text)
            else:
                # 默认尝试就业率数据，再试试流向数据
                data = self.data_extractor.extract_employment_rate_data(response.text)
                if not data or not data.get('major_data'):
                    data = self.data_extractor.extract_employment_flow_data(response.text)

            # 调试：如果提取到的数据为空或主要字段均为0，则保存页面以便排查
            try:
                ts = int(time.time())
                need_save = False
                if not data:
                    need_save = True
                else:
                    if data.get('data_type') == 'employment_rate':
                        majors = data.get('major_data', [])
                        if not majors:
                            need_save = True
                        else:
                            # 只有当所有记录既无就业率、也无人数字段时才判为可疑
                            # 否则允许只含 student_number（基本统计）而无 employment_rate 的页面
                            if all((not m.get('employment_rate') and not m.get('graduate_number') and not m.get('student_number'))
                                   for m in majors):
                                need_save = True
                    elif data.get('data_type') == 'employment_flow':
                        inds = data.get('industry_data', [])
                        if not inds:
                            need_save = True
                        else:
                            if all((not i.get('flow_number') and not i.get('flow_rate')) for i in inds):
                                need_save = True

                if need_save:
                    safe_name = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff_-]", '_', school_name)[:50]
                    debug_file = f"debug_report_{safe_name}_{ts}.html"
                    with open(debug_file, 'w', encoding='utf-8') as f:
                        f.write(response.text)
                    logger.warning(f"crawl_data_screen: extracted suspicious/empty data, saved HTML to {debug_file}")
            except Exception as e:
                logger.debug(f"crawl_data_screen: 无法保存调试页面: {e}")

            return data
                
        except Exception as e:
            logger.error(f"爬取数据大屏失败: {e}")
            return None
    
    def flatten_employment_data(self, employment_data):
        """扁平化就业数据"""
        flattened = []
        
        if not employment_data:
            return flattened
            
        # 基础信息
        base_info = {
            'school_name': employment_data.get('school_name', ''),
            'year': employment_data.get('year', ''),
            'data_type': 'employment_rate'
        }
        
        # 专业数据
        major_data = employment_data.get('major_data', [])
        for item in major_data:
            row = base_info.copy()
            row.update({
                'education': item.get('education', ''),
                'major': item.get('major', ''),
                'graduate_number': item.get('graduate_number', 0),
                'employment_number': item.get('employment_number', 0),
                'employment_rate': item.get('employment_rate', 0)
            })
            flattened.append(row)
        
        return flattened
    
    def flatten_basic_stats_data(self, stats_data):
        """扁平化基本统计数据"""
        flattened = []
        
        if not stats_data:
            return flattened
            
        # 基础信息
        base_info = {
            'school_name': stats_data.get('school_name', ''),
            'year': stats_data.get('year', ''),
            'data_type': 'basic_stats'
        }
        
        # 专业学生数据
        major_data = stats_data.get('major_data', [])
        for item in major_data:
            row = base_info.copy()
            row.update({
                'education': item.get('education', ''),
                'major': item.get('major', ''),
                'student_number': item.get('student_number', 0),
                'graduate_number': item.get('graduate_number', 0),
                'employment_number': item.get('employment_number', 0),
                'employment_rate': item.get('employment_rate', None)
            })
            flattened.append(row)
        
        return flattened
    
    def flatten_flow_data(self, flow_data):
        """扁平化就业流向数据"""
        flattened = []
        
        if not flow_data:
            return flattened
            
        # 基础信息
        base_info = {
            'school_name': flow_data.get('school_name', ''),
            'year': flow_data.get('year', ''),
            'data_type': 'employment_flow'
        }
        
        # 行业数据
        industry_data = flow_data.get('industry_data', [])
        for item in industry_data:
            row = base_info.copy()
            row.update({
                'education': item.get('education', ''),
                'trade_name': item.get('trade_name', ''),
                'flow_number': item.get('flow_number', 0),
                'flow_rate': item.get('flow_rate', 0)
            })
            flattened.append(row)
        
        # 省份数据
        province_data = flow_data.get('province_data', [])
        for item in province_data:
            row = base_info.copy()
            row.update({
                'province': item.get('name', ''),
                'province_flow_number': item.get('value', 0)
            })
            flattened.append(row)
        
        return flattened
    
    def run_crawler(self, max_schools=5, get_employment_data=True):
        """运行完整的爬虫"""
        logger.info("开始爬取高校就业数据...")
        
        # 登录
        if not self.login():
            return []
        
        # 获取学校列表
        schools = self.get_school_list(max_pages=100)
        logger.info(f"成功获取 {len(schools)} 所学校的基本信息")
        
        if not schools:
            logger.error("未获取到任何学校数据")
            return []
        
        complete_data = []
        
        # 处理每所学校
        for i, school in enumerate(schools[:max_schools]):
            logger.info(f"处理第 {i+1}/{min(max_schools, len(schools))} 所学校: {school['school_name']}")
            
            try:
                school_data = school.copy()
                
                # 获取数据大屏链接
                if get_employment_data:
                    data_reports = self.get_school_data_screens(school)
                    if data_reports:
                        employment_data = []
                        for report in data_reports[:3]:  # 获取前3个报告年度
                            # 分别请求就业率和流向页面
                            for data_type, url in [('employment_rate', report['employment_rate_url']), 
                                                    ('employment_flow', report['employment_flow_url'])]:
                                data = self.crawl_data_screen(url, school['school_name'], report.get('title', ''))
                                if data:
                                    employment_data.append({
                                        'report_title': report['title'],
                                        'report_url': url,
                                        'data_type': data_type,
                                        'data': data
                                    })
                        school_data['employment_data'] = employment_data
                
                complete_data.append(school_data)
                logger.info(f"✓ 完成学校: {school['school_name']}")
                
                time.sleep(2)  # 礼貌延时
                
            except Exception as e:
                logger.error(f"✗ 处理学校 {school['school_name']} 时出错: {e}")
                continue
        
        self.all_data = complete_data
        return complete_data
    
    def save_comprehensive_data(self, filename='高校就业综合数据.xlsx'):
        """保存综合数据到Excel"""
        if not self.all_data:
            logger.error("没有数据可保存")
            return False
            
        try:
            # 分离不同类型的数据
            employment_rate_data = []
            employment_flow_data = []
            basic_stats_data = []
            
            for school_data in self.all_data:
                school_name = school_data['school_name']
                
                # 处理就业数据
                if school_data.get('employment_data'):
                    for employment_item in school_data['employment_data']:
                        report_data = employment_item.get('data', {})
                        
                        # 处理就业率数据
                        if report_data.get('data_type') == 'employment_rate':
                            flattened = self.flatten_employment_data(report_data)
                            employment_rate_data.extend(flattened)
                        
                        # 处理基本统计数据
                        elif report_data.get('data_type') == 'basic_stats':
                            flattened = self.flatten_basic_stats_data(report_data)
                            basic_stats_data.extend(flattened)
                        
                        # 处理就业流向数据
                        elif report_data.get('data_type') == 'employment_flow':
                            flattened = self.flatten_flow_data(report_data)
                            employment_flow_data.extend(flattened)
            
            # 至少要有一种数据
            if not employment_rate_data and not employment_flow_data and not basic_stats_data:
                logger.error("没有提取到有效数据")
                return False
            
            # 创建Excel写入器，使用多个sheet
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # 就业率数据表
                if employment_rate_data:
                    df = pd.DataFrame(employment_rate_data)
                    df.to_excel(writer, sheet_name='就业率数据', index=False)
                    logger.info(f"  就业率数据: {len(employment_rate_data)} 条")
                
                # 基本统计数据表
                if basic_stats_data:
                    df = pd.DataFrame(basic_stats_data)
                    df.to_excel(writer, sheet_name='基本统计数据', index=False)
                    logger.info(f"  基本统计数据: {len(basic_stats_data)} 条")
                
                # 就业流向数据表
                if employment_flow_data:
                    df = pd.DataFrame(employment_flow_data)
                    df.to_excel(writer, sheet_name='就业流向数据', index=False)
                    logger.info(f"  就业流向数据: {len(employment_flow_data)} 条")
            
            total_records = len(employment_rate_data) + len(basic_stats_data) + len(employment_flow_data)
            logger.info(f"✓ 成功保存 {total_records} 条数据到 {filename}")

            return True
            
        except Exception as e:
            logger.error(f"✗ 保存文件时出错: {e}")
            return False
    
    def generate_analysis_report(self):
        """生成数据分析报告"""
        if not self.all_data:
            return "没有数据可分析"
        
        report_lines = ["高校就业数据分析报告", "="*50]
        
        for school_data in self.all_data:
            school_name = school_data['school_name']
            report_lines.append(f"\n学校: {school_name}")
            report_lines.append("-"*30)
            
            # 分析就业数据
            if school_data.get('employment_data'):
                for employment_item in school_data['employment_data']:
                    report_title = employment_item.get('report_title', '')
                    data = employment_item.get('data', {})
                    year = data.get('year', '未知')
                    
                    report_lines.append(f"\n{year}届 - {report_title}:")
                    
                    # 专业数量
                    major_data = data.get('major_data', [])
                    if major_data:
                        report_lines.append(f"  专业数量: {len(major_data)}个")
                        # 显示前几个专业
                        for i, major in enumerate(major_data[:3]):
                            report_lines.append(f"    {major.get('major', '')}: {major.get('employment_rate', 0)}%")
                    
                    # 行业数据
                    industry_data = data.get('industry_data', [])
                    if industry_data:
                        report_lines.append(f"  行业分类: {len(industry_data)}个")
                        # 显示前几个行业
                        for i, industry in enumerate(industry_data[:3]):
                            report_lines.append(f"    {industry.get('trade_name', '')}: {industry.get('flow_rate', 0)}%")
        
        return "\n".join(report_lines)

# 简化版本的主函数，避免错误
def main():
    # 您的用户名和密码
    USERNAME = ""
    PASSWORD = ""
    
    # 创建爬虫实例
    crawler = AdvancedEmploymentCrawler(username=USERNAME, password=PASSWORD)
    
    # 运行爬虫（爬取所有学校）
    print("开始爬取数据...")
    data = crawler.run_crawler(max_schools=1500, get_employment_data=True)
    
    # 保存结果
    if data:
        success = crawler.save_comprehensive_data('高校就业详细数据.xlsx')
        if success:
            print("✓ 数据已保存到 '高校就业详细数据.xlsx'")
            
            # 显示一些统计信息
            total_records = 0
            for school in data:
                if school.get('employment_data'):
                    total_records += len(school['employment_data'])
            print(f"✓ 共获取 {len(data)} 所学校，{total_records} 个数据报告")
            
            # 尝试生成分析报告，如果出错则跳过
            try:
                report = crawler.generate_analysis_report()
                print("\n" + report)
                
                # 保存报告到文件
                with open('数据分析报告.txt', 'w', encoding='utf-8') as f:
                    f.write(report)
                print("✓ 分析报告已保存到 '数据分析报告.txt'")
            except Exception as e:
                print(f"⚠ 生成分析报告时出错: {e}")
                print("✓ 数据已成功保存，但分析报告生成失败")
        else:
            print("✗ 数据保存失败")
    else:
        print("✗ 爬取失败，未获取到数据")

if __name__ == "__main__":
    main()