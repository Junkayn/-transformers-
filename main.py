import os
import time
import pickle
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from transformers import pipeline
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.service import Service
from selenium.webdriver.edge.options import Options
from selenium.common.exceptions import NoSuchElementException

# ------------------------------- 中文显示设置 -------------------------------
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 中文字体
plt.rcParams['axes.unicode_minus'] = False

# ------------------------------- 初始化情绪分析模型 -------------------------------
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment"
)

# ------------------------------- 浏览器初始化 -------------------------------
def init_browser():
    edge_driver_path = os.path.expanduser(r"D:\ljz-privated\python\analyze\msedgedriver.exe")  # 改成你的路径
    service = Service(edge_driver_path)
    options = Options()
    options.add_argument("--start-maximized")
    driver = webdriver.Edge(service=service, options=options)
    return driver

# ------------------------------- Cookies -------------------------------
def save_cookies(driver, cookies_file="cookies.pkl"):
    cookies = driver.get_cookies()
    with open(cookies_file, "wb") as f:
        pickle.dump(cookies, f)

def load_cookies(driver, cookies_file="cookies.pkl"):
    if os.path.exists(cookies_file):
        driver.get("https://weibo.com/")
        time.sleep(2)
        with open(cookies_file, "rb") as f:
            cookies = pickle.load(f)
            for cookie in cookies:
                driver.add_cookie(cookie)
        driver.get("https://weibo.com/")
        time.sleep(2)
        return True
    return False

# ------------------------------- 等待登录 -------------------------------
def wait_for_login(driver, progress_callback=None):
    driver.get("https://weibo.com/login.php")
    if progress_callback:
        progress_callback("请在浏览器中手动登录微博…")
    time.sleep(5)
    last_url = ""
    while True:
        time.sleep(3)
        current_url = driver.current_url
        if "login" not in current_url and "passport" not in current_url:
            try:
                driver.find_element(By.CSS_SELECTOR, "input[placeholder='搜索微博']")
                save_cookies(driver)
                if progress_callback:
                    progress_callback("✅ 登录成功")
                break
            except NoSuchElementException:
                pass
        if current_url != last_url:
            last_url = current_url
    # 如果长时间没登录，也可以在这里设置超时提示登录失败（可选）

# ------------------------------- 抓取评论 -------------------------------
def get_comments(driver, url, max_comments=200, progress_callback=None):
    driver.get(url)
    time.sleep(5)
    users, comments = [], []

    while len(comments) < max_comments:
        user_elems = driver.find_elements(By.CSS_SELECTOR, "div.con1.woo-box-item-flex a")
        comment_elems = driver.find_elements(By.CSS_SELECTOR, "div.text > span")

        for i in range(min(len(user_elems), len(comment_elems))):
            if len(comments) >= max_comments:
                break
            user = user_elems[i].text.strip()
            comment = comment_elems[i].text.strip()
            
            # --------- 过滤折叠回复 ---------
            if not comment or comment.startswith("共") and "条回复" in comment:
                continue

            users.append(user)
            comments.append(comment)

        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)

        if progress_callback:
            progress_callback(f"已抓取 {len(comments)} 条评论…")

        # 当已经没有新评论加载时，退出循环
        if len(user_elems) == 0 or len(comment_elems) == 0:
            break

    return users, comments


# ------------------------------- 情绪分析 -------------------------------
def analyze_sentiment(users, comments, progress_callback=None):
    data = []
    for idx, (user, c) in enumerate(zip(users, comments), start=1):
        try:
            result = sentiment_pipeline(c[:512])[0]
            label = result['label']
            score = result['score']

            threshold = 0.6
            if score < threshold:
                sentiment = "中性"
            elif label == 'Positive':
                sentiment = "积极"
            elif label == 'Negative':
                sentiment = "消极"
            else:
                sentiment = "中性"
        except:
            sentiment = "中性"
            score = 0
        data.append({"用户": user, "评论": c, "情绪": sentiment, "置信度": score})

        if progress_callback:
            progress_callback(f"分析评论 {idx}/{len(comments)}")

    return pd.DataFrame(data)

# ------------------------------- 可视化 -------------------------------
def visualize_sentiment(df, canvas_frame):
    summary = df["情绪"].value_counts(normalize=True) * 100
    fig, ax = plt.subplots(figsize=(5,3))
    summary.plot(kind="bar", ax=ax, color=["green","red","gray"])
    ax.set_title("情绪分布")
    ax.set_ylabel("百分比")

    for i, v in enumerate(summary):
        ax.text(i, v + 0.5, f"{v:.1f}%", ha='center', fontname='Microsoft YaHei')

    plt.tight_layout()
    canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill='both', expand=True)

# ------------------------------- Tkinter UI -------------------------------
class App:
    def __init__(self, root):
        self.root = root
        root.title("微博评论情绪分析")
        root.geometry("600x500")

        self.label = tk.Label(root, text="请输入要分析的微博帖子链接：")
        self.label.pack(pady=5)

        self.entry = tk.Entry(root, width=80)
        self.entry.pack(pady=5)

        self.progress_label = tk.Label(root, text="")
        self.progress_label.pack(pady=5)

        self.start_btn = tk.Button(root, text="开始分析", command=self.start_analysis)
        self.start_btn.pack(pady=5)

        self.canvas_frame = tk.Frame(root)
        self.canvas_frame.pack(fill='both', expand=True)

    def update_progress(self, msg):
        self.progress_label.config(text=msg)
        self.root.update()

    def start_analysis(self):
        weibo_url = self.entry.get().strip()
        if not weibo_url:
            messagebox.showwarning("提示", "请输入微博帖子链接")
            return

        self.update_progress("初始化浏览器…")
        driver = init_browser()

        # 登录处理
        if not load_cookies(driver):
            wait_for_login(driver, progress_callback=self.update_progress)
        else:
            self.update_progress("✅ 已加载登录状态")

        # 抓取评论
        self.update_progress("开始抓取评论…")
        users, comments = get_comments(driver, weibo_url, max_comments=200, progress_callback=self.update_progress)

        driver.quit()

        if len(comments) == 0:
            messagebox.showerror("错误", "未抓取到评论，请检查微博是否公开或评论区是否加载。")
            return

        # 情绪分析
        self.update_progress("开始情绪分析…")
        df = analyze_sentiment(users, comments, progress_callback=self.update_progress)
        df.to_csv("weibo_comments_sentiment.csv", index=False, encoding="utf-8-sig")

        self.update_progress("分析完成，已生成 CSV 文件")
        visualize_sentiment(df, self.canvas_frame)

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
