from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import json
import pandas as pd

def save_cookies(driver, filename="weibo_cookies.json"):
    """保存cookies到文件"""
    cookies = driver.get_cookies()
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(cookies, f, ensure_ascii=False, indent=2)
    print("Cookies已保存")

def load_cookies(driver, filename="weibo_cookies.json"):
    """从文件加载cookies"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            cookies = json.load(f)
        
        driver.get("https://weibo.com")  # 先访问域名
        time.sleep(2)
        
        for cookie in cookies:
            # 确保cookie有必要的字段
            if 'expiry' in cookie:
                # 移除expiry字段或转换为int
                cookie['expiry'] = int(cookie['expiry'])
            driver.add_cookie(cookie)
        
        print("Cookies已加载")
        return True
    except FileNotFoundError:
        print("未找到cookies文件")
        return False

def manual_login_and_save_cookies(driver):
    """手动登录并保存cookies"""
    print("请手动登录微博...")
    driver.get("https://weibo.com/login")
    
    # 等待用户手动登录
    input("请在浏览器中登录微博，登录完成后按Enter键继续...")
    
    # 保存cookies
    save_cookies(driver)
    return True

# 用户自定义设置
keyword = "我在零售业工作，我的职场心态"
max_posts = 5000  # 设置最大爬取数量

# 主程序
options = webdriver.ChromeOptions()
options.add_argument("--disable-blink-features=AutomationControlled")
driver = webdriver.Chrome(options=options)

# 尝试加载cookies
if not load_cookies(driver):
    # 如果cookies不存在或无效，进行手动登录
    manual_login_and_save_cookies(driver)

# 刷新页面使cookies生效
driver.get("https://weibo.com")
time.sleep(3)

# 检查是否登录成功
try:
    wait = WebDriverWait(driver, 10)
    wait.until(
        EC.presence_of_element_located((By.CSS_SELECTOR, ".gn_name"))
    )
    print("登录状态验证成功！")
except:
    print("登录状态可能失效，请重新登录")
    if manual_login_and_save_cookies(driver):
        driver.get("https://weibo.com")
        time.sleep(3)

# 开始爬取数据
print(f"开始爬取关键词: {keyword}")
print(f"目标数量: {max_posts}条")
driver.get(f"https://s.weibo.com/weibo?q={keyword}")

# 等待页面加载
wait = WebDriverWait(driver, 10)
try:
    wait.until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "div.card-wrap"))
    )
except:
    print("等待搜索结果超时，继续执行...")

# 模拟下拉滚动多次，添加最大数量控制
data = []
scroll_count = 0
max_scrolls = 20  # 最大滚动次数，防止无限滚动
page_num = 1  # 当前页码

print("\n开始爬取，请稍候...")
print("-" * 50)

while len(data) < max_posts:
    # 滚动页面
    for i in range(5):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
    
    # 获取当前页面的微博
    posts = driver.find_elements(By.CSS_SELECTOR, "div.card-wrap")
    
    new_posts_count = 0
    for post in posts:
        try:
            text = post.find_element(By.CSS_SELECTOR, ".content").text
            # 去重检查
            if text not in [d[2] for d in data]:
                user = post.find_element(By.CSS_SELECTOR, ".name").text
                timeinfo = post.find_element(By.CSS_SELECTOR, ".from").text
                data.append([user, timeinfo, text])
                new_posts_count += 1
                
                # 前台显示进度
                print(f"已收集: {len(data)}/{max_posts}条", end='\r')
                
                # 达到最大数量就停止
                if len(data) >= max_posts:
                    print(f"已达到目标数量: {max_posts}条")
                    break
        except:
            continue
    
    scroll_count += 1
    
    # 显示滚动进度
    if scroll_count % 5 == 0:
        print(f"已滚动 {scroll_count} 次，收集到 {len(data)} 条微博")
    
    # 翻页功能 - 只增加这一部分
    if len(data) < max_posts:
        try:
            # 尝试点击下一页
            next_button = driver.find_element(By.CSS_SELECTOR, "a.next")
            if next_button.is_displayed():
                print(f"正在翻页到第 {page_num + 1} 页...")
                next_button.click()
                time.sleep(3)  # 等待新页面加载
                page_num += 1
                scroll_count = 0  # 重置滚动计数
            else:
                print("已到达最后一页")
                break
        except:
            print("无法翻页，可能已到最后一页")
            break

print(f"\n爬取完成！共收集到 {len(data)} 条微博")
print("-" * 50)

driver.quit()

# 保存数据
if data:
    # 去重（虽然上面已经做了，这里再确保一次）
    unique_data = []
    seen = set()
    for item in data:
        if item[2] not in seen:
            seen.add(item[2])
            unique_data.append(item)
    
    df = pd.DataFrame(unique_data, columns=["用户", "时间", "内容"])
    filename = f"selenium_微博数据_{keyword}_{len(unique_data)}条.csv"
    df.to_csv(filename, index=False, encoding="utf-8-sig")
    print(f"数据已保存到: {filename}")
    print(f"实际有效数据: {len(unique_data)} 条（已去重）")
else:
    print("未收集到任何数据")