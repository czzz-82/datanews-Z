import requests
import pandas as pd
import time
import random
from bs4 import BeautifulSoup
import datetime
import json
import os
from typing import List, Dict

class CookieManager:
    """Cookie池管理器"""
    def __init__(self, cookie_file="cookie_pool.json"):
        self.cookie_file = cookie_file
        self.cookie_pool = []
        self.current_index = 0
        self.load_cookies()
    
    def load_cookies(self):
        """从文件加载Cookie池"""
        if os.path.exists(self.cookie_file):
            try:
                with open(self.cookie_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.cookie_pool = data.get("cookies", [])
                print(f"已加载 {len(self.cookie_pool)} 个Cookie")
            except Exception as e:
                print(f"加载Cookie文件失败: {e}")
                self.cookie_pool = []
        else:
            print(f"Cookie文件 {self.cookie_file} 不存在")
            self.cookie_pool = []
    
    def save_cookies(self):
        """保存Cookie池到文件"""
        try:
            with open(self.cookie_file, 'w', encoding='utf-8') as f:
                json.dump({"cookies": self.cookie_pool}, f, ensure_ascii=False, indent=2)
            print(f"Cookie池已保存，当前共有 {len(self.cookie_pool)} 个Cookie")
        except Exception as e:
            print(f"保存Cookie文件失败: {e}")
    
    def add_cookie(self, cookie_str: str, description: str = ""):
        """添加新的Cookie"""
        cookie_item = {
            "cookie": cookie_str,
            "description": description,
            "last_used": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "success_count": 0,
            "fail_count": 0,
            "status": "active"
        }
        self.cookie_pool.append(cookie_item)
        self.save_cookies()
        print(f"已添加Cookie: {description}")
        return len(self.cookie_pool) - 1  # 返回新Cookie的索引
    
    def get_current_cookie(self) -> str:
        """获取当前Cookie"""
        if not self.cookie_pool:
            raise ValueError("Cookie池为空，请添加Cookie")
        
        # 获取当前Cookie
        cookie_item = self.cookie_pool[self.current_index]
        cookie_item["last_used"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return cookie_item["cookie"]
    
    def rotate_cookie(self):
        """切换到下一个Cookie"""
        if not self.cookie_pool:
            return False
        
        old_index = self.current_index
        self.current_index = (self.current_index + 1) % len(self.cookie_pool)
        
        if old_index != self.current_index:
            print(f"已切换Cookie: {old_index} -> {self.current_index}")
            return True
        return False
    
    def set_current_cookie(self, index: int):
        """设置当前使用的Cookie索引"""
        if 0 <= index < len(self.cookie_pool):
            self.current_index = index
            return True
        return False
    
    def mark_success(self):
        """标记当前Cookie使用成功"""
        if self.cookie_pool:
            self.cookie_pool[self.current_index]["success_count"] += 1
    
    def mark_fail(self):
        """标记当前Cookie使用失败"""
        if self.cookie_pool:
            self.cookie_pool[self.current_index]["fail_count"] += 1
    
    def remove_cookie(self, index: int):
        """移除指定索引的Cookie"""
        if 0 <= index < len(self.cookie_pool):
            removed = self.cookie_pool.pop(index)
            if index < self.current_index:
                self.current_index -= 1
            elif index == self.current_index and self.current_index >= len(self.cookie_pool):
                self.current_index = 0
            print(f"已移除Cookie: {removed.get('description', '未知')}")
            self.save_cookies()
    
    def get_status(self) -> Dict:
        """获取Cookie池状态"""
        return {
            "total": len(self.cookie_pool),
            "current_index": self.current_index,
            "active_count": len([c for c in self.cookie_pool if c.get("status") == "active"]),
            "cookies": self.cookie_pool
        }

class BossSpiderAPI:
    def __init__(self):
        self.base_url = "https://www.zhipin.com/wapi/zpgeek/search/joblist.json"
        self.cookie_manager = CookieManager()
        
        # 初始化headers
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36",
            "Referer": "https://www.zhipin.com/",
            "Cookie": ""  # 初始为空，会从Cookie池获取
        }
        
        self.params = {
            "city": "101020100",  
            "pageSize": 100
        }
        
        self.data_list = []
        self.max_retry = 3  # 最大重试次数
        self.empty_page_count = 0  # 连续空页计数器
        self.max_empty_pages = 1  # 最大允许连续空页数
        
        # 尝试获取初始Cookie
        self.update_headers_cookie()
    
    def update_headers_cookie(self):
        """更新headers中的Cookie"""
        try:
            cookie_str = self.cookie_manager.get_current_cookie()
            self.headers["Cookie"] = cookie_str
            # 同时更新detail_headers
            self.detail_headers = {
                "User-Agent": self.headers["User-Agent"],
                "Cookie": cookie_str
            }
        except ValueError as e:
            print(f"Cookie获取失败: {e}")
    
    def fetch_data_with_retry(self, max_pages=3):
        """带重试机制的数据抓取"""
        page = 1
        retry_count = 0
        
        while page <= max_pages:
            print(f"\n{'='*50}")
            print(f"尝试抓取第 {page} 页 (Cookie #{self.cookie_manager.current_index})")
            
            self.params['page'] = page
            
            try:
                resp = requests.get(self.base_url, headers=self.headers, params=self.params, timeout=10)
                resp.raise_for_status()
                
                # 检查响应状态
                if resp.status_code == 403:
                    print("403 Forbidden: Cookie可能失效")
                    self.cookie_manager.mark_fail()
                    if not self.handle_cookie_failure():
                        break
                    continue
                
                result = resp.json()
                job_list = result.get("zpData", {}).get("jobList", [])
                
                if not job_list:
                    self.empty_page_count += 1
                    print(f"返回的jobList为空 (连续空页: {self.empty_page_count}/{self.max_empty_pages})")
                    
                    if self.empty_page_count >= self.max_empty_pages:
                        print(f"连续 {self.empty_page_count} 页获取空数据，可能Cookie已失效")
                        
                        # 询问用户是否添加新Cookie
                        add_new = input("\n检测到Cookie可能失效，是否输入新Cookie？(y/n): ").strip().lower()
                        if add_new == 'y':
                            if self.add_cookie_interactive():
                                # 重置空页计数器
                                self.empty_page_count = 0
                                # 重新尝试当前页
                                continue
                            else:
                                print("未添加新Cookie，继续使用当前Cookie...")
                                retry_count += 1
                        else:
                            print("继续使用当前Cookie...")
                            retry_count += 1
                    else:
                        retry_count += 1
                    
                    if retry_count >= self.max_retry:
                        print(f"连续 {retry_count} 次获取空数据，尝试切换Cookie...")
                        self.cookie_manager.mark_fail()
                        
                        if not self.handle_cookie_failure():
                            print("所有Cookie都已尝试，停止抓取")
                            break
                        retry_count = 0
                        self.empty_page_count = 0
                        continue
                    
                    print(f"等待后重试... (第{retry_count}次)")
                    time.sleep(random.uniform(2, 3))
                    continue
                
                # 成功获取数据，重置计数器
                self.empty_page_count = 0
                self.cookie_manager.mark_success()
                retry_count = 0
                
                # 处理数据
                self.process_job_list(job_list, page)
                page += 1
                
                # 随机延迟，避免请求过快
                time.sleep(random.uniform(1, 2))
                
            except requests.RequestException as e:
                print(f"请求失败：{e}")
                self.cookie_manager.mark_fail()
                
                if not self.handle_cookie_failure():
                    break
                retry_count = 0
                self.empty_page_count = 0
                
            except Exception as e:
                print(f"处理数据时发生错误：{e}")
                break
    
    def handle_cookie_failure(self) -> bool:
        """
        处理Cookie失效情况
        返回: True - 成功切换Cookie并继续; False - 无法继续
        """
        print("检测到Cookie可能失效，尝试切换...")
        
        # 尝试切换Cookie
        if self.cookie_manager.rotate_cookie():
            self.update_headers_cookie()
            print(f"已切换到Cookie #{self.cookie_manager.current_index}")
            time.sleep(random.uniform(2, 4))  # 切换后等待更长时间
            return True
        else:
            print("Cookie池中只有一个Cookie或无法切换")
            
            # 询问是否添加新Cookie
            add_new = input("是否输入新Cookie？(y/n): ").strip().lower()
            if add_new == 'y':
                return self.add_cookie_interactive()
            else:
                print("使用当前Cookie继续...")
                return True
    
    def add_cookie_interactive(self) -> bool:
        """交互式添加Cookie"""
        print("\n" + "="*50)
        print("请输入新的Cookie：")
        print("（获取方法：登录BOSS直聘后，按F12打开开发者工具，")
        print("在Network标签中找到任意请求，复制Request Headers中的Cookie字段）")
        print("="*50)
        
        cookie_str = input("\n请输入Cookie字符串: ").strip()
        if not cookie_str:
            print("Cookie为空，取消添加")
            return False
        
        description = input("请输入Cookie描述(可选): ").strip() or f"手动添加_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 添加新Cookie
        new_index = self.cookie_manager.add_cookie(cookie_str, description)
        
        # 切换到新添加的Cookie
        self.cookie_manager.set_current_cookie(new_index)
        self.update_headers_cookie()
        
        print(f"已添加并切换到新Cookie #{new_index}")
        
        # 测试新Cookie是否有效
        print("正在测试新Cookie...")
        try:
            test_resp = requests.get(self.base_url, headers=self.headers, params={"page": 1, "pageSize": 10}, timeout=10)
            if test_resp.status_code == 200:
                result = test_resp.json()
                job_list = result.get("zpData", {}).get("jobList", [])
                if job_list:
                    print(f"Cookie测试成功，获取到 {len(job_list)} 个职位")
                    return True
                else:
                    print("Cookie测试：获取到空数据，可能仍需验证")
                    return True
            else:
                print(f"Cookie测试失败，状态码: {test_resp.status_code}")
                return False
        except Exception as e:
            print(f"Cookie测试异常: {e}")
            return True  # 仍然尝试使用
    
    def process_job_list(self, job_list, page):
        """处理职位列表数据"""
        print(f"第 {page} 页获取到 {len(job_list)} 个职位")
        
        for i, job in enumerate(job_list):
            job_id = job.get("encryptJobId")
            print(f"  处理职位 {i+1}/{len(job_list)}: {job.get('jobName')}")
            
            job_desc = self.get_job_detail(job_id)
            
            item = {
                "职位": job.get("jobName"),
                "公司": job.get("brandName"),
                "薪资": job.get("salaryDesc"),
                "地区": job.get("cityName"),
                "经验": job.get("jobExperience"),
                "学历": job.get("jobDegree"),
                "公司规模": job.get("brandScaleName"),
                "行业": job.get("brandIndustry"),
                "福利标签": ",".join(job.get("welfareList", [])),
                "技能标签": ",".join(job.get("skills", [])),
                "职位描述": job_desc,
                "抓取时间": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "使用的Cookie索引": self.cookie_manager.current_index
            }
            self.data_list.append(item)
            
            # 详情页请求间隔
            time.sleep(random.uniform(0.5, 1.5))
    
    def get_job_detail(self, job_id):
        """获取职位详情"""
        url = f"https://www.zhipin.com/job_detail/{job_id}.html"
        try:
            resp = requests.get(url, headers=self.detail_headers, timeout=10)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, 'html.parser')
                desc_tag = soup.select_one('.job-sec-text')
                return desc_tag.text.strip() if desc_tag else ""
            else:
                return f"详情页状态码: {resp.status_code}"
        except Exception as e:
            return f"详情页获取失败: {str(e)}"
    
    def save_excel(self):
        """保存数据到Excel"""
        if not self.data_list:
            print("没有数据需要保存")
            return
        
        filename = f"武汉岗位数据_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        df = pd.DataFrame(self.data_list)
        df.to_excel(filename, index=False)
        print(f"保存成功：{filename}，共 {len(df)} 条职位")
        
        # 同时保存一份CSV备份
        csv_filename = f"上海岗位数据_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
        print(f"CSV备份：{csv_filename}")
    
    def add_cookie_manually(self):
        """手动添加Cookie（菜单功能）"""
        print("\n手动添加Cookie")
        print("="*50)
        cookie_str = input("请输入Cookie字符串: ").strip()
        if not cookie_str:
            print("Cookie为空，取消添加")
            return
        
        description = input("请输入Cookie描述(可选): ").strip() or "手动添加"
        
        self.cookie_manager.add_cookie(cookie_str, description)
        print("Cookie添加成功！")
        
        # 更新当前使用的Cookie
        self.update_headers_cookie()
    
    def view_cookie_status(self):
        """查看Cookie状态"""
        status = self.cookie_manager.get_status()
        print("\nCookie池状态:")
        print(f"总数: {status['total']}")
        print(f"当前使用: #{status['current_index']}")
        print(f"活跃数: {status['active_count']}")
        
        print("\nCookie详情:")
        for i, cookie_item in enumerate(status['cookies']):
            current_mark = " ← 当前使用" if i == status['current_index'] else ""
            print(f"  [{i}] {cookie_item.get('description', '无描述')}{current_mark}")
            print(f"     成功: {cookie_item.get('success_count', 0)}, 失败: {cookie_item.get('fail_count', 0)}")
            print(f"     最后使用: {cookie_item.get('last_used', '从未使用')}")
    
    def run(self, max_pages=3):
        """运行爬虫"""
        print("BOSS直聘爬虫启动")
        print("="*50)
        
        # 检查Cookie池
        status = self.cookie_manager.get_status()
        if status['total'] == 0:
            print("警告: Cookie池为空！")
            choice = input("是否现在添加Cookie? (y/n): ").strip().lower()
            if choice == 'y':
                self.add_cookie_interactive()
            else:
                print("无法运行，需要至少一个Cookie")
                return
        
        self.view_cookie_status()
        
        # 开始抓取
        self.fetch_data_with_retry(max_pages)
        
        # 保存数据
        self.save_excel()
        
        # 显示最终状态
        print("\n" + "="*50)
        print("抓取完成！")
        self.view_cookie_status()

def create_cookie_file_example():
    """创建Cookie文件示例"""
    example_data = {
        "cookies": [
            {
                "cookie": "lastCity=101200100; ab_guid=3933e957-a8b8-46f6-93b0-90ad02b39640; __g=sem_bingpc; ...",
                "description": "用户A的Cookie",
                "last_used": None,
                "success_count": 0,
                "fail_count": 0,
                "status": "active"
            }
        ]
    }
    
    with open("cookie_pool.json", "w", encoding="utf-8") as f:
        json.dump(example_data, f, ensure_ascii=False, indent=2)
    
    print("已创建示例Cookie文件: cookie_pool.json")
    print("请将您的Cookie复制到该文件中，或运行爬虫时在终端输入")

if __name__ == "__main__":
    # 创建示例Cookie文件（如果不存在）
    if not os.path.exists("cookie_pool.json"):
        print("检测到没有cookie_pool.json文件")
        choice = input("是否创建示例文件? (y/n): ").strip().lower()
        if choice == 'y':
            create_cookie_file_example()
            print("\n建议：")
            print("1. 将您的Cookie添加到cookie_pool.json文件中，或")
            print("2. 运行爬虫时，当检测到Cookie失效时在终端输入新Cookie")
        else:
            print("将使用空Cookie池开始运行")
    
    # 运行爬虫
    spider = BossSpiderAPI()
    
    # 提供交互菜单
    while True:
        print("\n" + "="*50)
        print("BOSS直聘爬虫菜单")
        print("="*50)
        print("1. 开始抓取数据")
        print("2. 查看Cookie状态")
        print("3. 添加新Cookie")
        print("4. 退出程序")
        
        choice = input("\n请选择操作 (1-4): ").strip()
        
        if choice == '1':
            max_pages = input("请输入要抓取的页数 (默认3): ").strip()
            max_pages = int(max_pages) if max_pages.isdigit() else 3
            spider.run(max_pages)
        elif choice == '2':
            spider.view_cookie_status()
        elif choice == '3':
            spider.add_cookie_manually()
        elif choice == '4':
            print("退出程序")
            break
        else:
            print("无效选择，请重新输入")