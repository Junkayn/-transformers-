import os
import time
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.service import Service
from selenium.webdriver.edge.options import Options
from selenium.common.exceptions import NoSuchElementException

# -------------------------------
# 初始化 Edge 浏览器
# -------------------------------
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment"  # 更换成三分类情绪模型
)

def init_browser():
    edge_driver_path = os.path.expanduser(r"D:\ljz-privated\python\analyze\msedgedriver.exe")  # ← 修改成你的路径
    service = Service(edge_driver_path)
    options = Options()
    options.add_argument("--start-maximized")
    driver = webdriver.Edge(service=service, options=options)
    return driver

# -------------------------------
# 保存 Cookies
# -------------------------------
def save_cookies(driver, cookies_file="cookies.pkl"):
    cookies = driver.get_cookies()
    with open(cookies_file, "wb") as f:
        pickle.dump(cookies, f)
    print("✅ 登录状态已保存。")

# -------------------------------
# 加载 Cookies
# -------------------------------
def load_cookies(driver, cookies_file="cookies.pkl"):
    if os.path.exists(cookies_file):
        driver.get("https://weibo.com/")
        time.sleep(2)  # 等待页面加载
        with open(cookies_file, "rb") as f:
            cookies = pickle.load(f)
            for cookie in cookies:
                driver.add_cookie(cookie)
        driver.get("https://weibo.com/")  # 刷新页面以应用 cookies
        time.sleep(2)
        print("✅ 登录状态已加载。")
        return True  # 成功加载 cookies
    else:
        print("⚠️ 未找到保存的登录 cookies，需要重新登录。")
        return False  # 没有找到 cookies，需要重新登录

# -------------------------------
# 等待用户登录微博（新版适配）
# -------------------------------
def wait_for_login(driver):
    driver.get("https://weibo.com/login.php")
    print("请在打开的浏览器中手动登录微博...")
    time.sleep(5)
    last_url = ""

    while True:
        time.sleep(3)
        current_url = driver.current_url
        if "login" not in current_url and "passport" not in current_url:
            try:
                driver.find_element(By.CSS_SELECTOR, "input[placeholder='搜索微博']")
                print("✅ 检测到微博主界面元素，登录成功！")
                save_cookies(driver)  # 登录成功后保存 cookies
                break
            except NoSuchElementException:
                pass
        if current_url != last_url:
            print(f"当前页面: {current_url}")
            last_url = current_url

# -------------------------------
# 抓取评论（新版结构）
# -------------------------------
def get_comments(driver, url, max_pages=5):
    driver.get(url)
    time.sleep(5)
    users, comments = [], []

    for page in range(1, max_pages + 1):
        print(f"正在抓取第 {page} 页评论… (当前已有 {len(comments)} 条)")
        time.sleep(3)

        user_elems = driver.find_elements(By.CSS_SELECTOR, "div.con1.woo-box-item-flex a")
        comment_elems = driver.find_elements(By.CSS_SELECTOR, "div.text > span")

        print(f"找到 {len(user_elems)} 个用户名元素，{len(comment_elems)} 条评论元素")

        for i in range(min(len(user_elems), len(comment_elems))):
            user = user_elems[i].text.strip()
            comment = comment_elems[i].text.strip()
            if comment:
                users.append(user)
                comments.append(comment)

        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(4)

    print(f"共抓取到 {len(comments)} 条评论。")
    return users, comments

# -------------------------------
# 情绪分析
# -------------------------------
def analyze_sentiment(users, comments):
    data = []
    for user, c in zip(users, comments):
        try:
            result = sentiment_pipeline(c[:512])[0]  # 截断防止过长
            label = result['label']
            score = result['score']
            
            print(f"评论: {c} -> {label}, 置信度: {score:.2f}")
            
            threshold = 0.6
            if score < threshold:
                sentiment = "中性"
            elif label == 'Positive':
                sentiment = "积极"
            elif label == 'Negative':
                sentiment = "消极"
            else:
                sentiment = "中性"

        except Exception as e:
            sentiment = "中性"
            print(f"Error: {e}")
        
        data.append({"用户": user, "评论": c, "情绪": sentiment, "置信度": score})
    
    return pd.DataFrame(data)

# -------------------------------
# 主程序
# -------------------------------
def main():
    driver = init_browser()

    # 先加载 cookies（如果已登录）
    if not load_cookies(driver):
        wait_for_login(driver)  # 如果 cookies 无效或不存在，执行登录

    weibo_url = input("请输入要分析的微博帖子链接：").strip()
    users, comments = get_comments(driver, weibo_url)

    if len(comments) == 0:
        print("⚠️ 未抓取到评论，请检查微博是否公开或评论区是否加载。")
        driver.quit()
        return

    df = analyze_sentiment(users, comments)
    print(df.head())

    df.to_csv("weibo_comments_sentiment.csv", index=False, encoding="utf-8-sig")
    print("✅ 已保存到 weibo_comments_sentiment.csv")

    summary = df["情绪"].value_counts(normalize=True) * 100
    print("\n总体情绪分布：")
    print(summary.round(2).to_string())

    driver.quit()

if __name__ == "__main__":
    main()
