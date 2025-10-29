import os
import time
import pickle
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import threading
import logging
import asyncio
from playwright.async_api import async_playwright

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_OFFLINE"] = "1"  
logging.getLogger("transformers").setLevel(logging.ERROR)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "Sentiment", "snapshots",
                          "7c257c5cde3225d0789acfa8d67eb043289b0295")
MODEL_CACHE = os.path.join(BASE_DIR, "model_cache.pkl")
COOKIES_PATH = os.path.join(BASE_DIR, "cookies.json")



async def async_browser_workflow(weibo_url, progress_callback):
    """异步执行微博登录、评论抓取"""
    async with async_playwright() as p:
        # 加载 cookies
        if os.path.exists(COOKIES_PATH):
            context = await p.chromium.launch_persistent_context(
                user_data_dir=os.path.join(BASE_DIR, "user_data"),
                channel="msedge",
                headless=False,
                storage_state=COOKIES_PATH
            )
            page = await context.new_page()
            await page.goto("https://weibo.com/")
            progress_callback("已加载登录状态✅️")
        else:
            browser = await p.chromium.launch(channel="msedge", headless=False)
            context = await browser.new_context()
            page = await context.new_page()
            await page.goto("https://weibo.com/login.php")
            progress_callback("请在浏览器中登录微博…")

            # 等待登录成功
            while True:
                await asyncio.sleep(3)
                url = page.url
                if "login" not in url and "passport" not in url:
                    progress_callback("登录成功✅️")
                    await context.storage_state(path=COOKIES_PATH)
                    break

        # 抓取评论
        progress_callback("开始抓取评论…")
        await page.goto(weibo_url)
        await asyncio.sleep(5)

        users, comments = [], []
        while len(comments) < 200:
            user_elems = await page.locator("div.con1.woo-box-item-flex a").all_text_contents()
            comment_elems = await page.locator("div.text > span").all_text_contents()

            for u, c in zip(user_elems, comment_elems):
                c = c.strip()
                if not c or (c.startswith("共") and "条回复" in c):
                    continue
                users.append(u.strip())
                comments.append(c)
                if len(comments) >= 200:
                    break

            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await asyncio.sleep(2)
            progress_callback(f"已抓取 {len(comments)} 条评论…")

            if len(user_elems) == 0 or len(comment_elems) == 0:
                break

        await context.close()
        return users, comments


def load_model(progress_callback=None):
    if progress_callback:
        progress_callback("正在加载情绪分析模型…")

    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, logging
    logging.set_verbosity_error()

    if os.path.exists(MODEL_CACHE):
        try:
            if progress_callback:
                progress_callback("加载本地模型缓存中…")
            with open(MODEL_CACHE, "rb") as f:
                tokenizer, model = pickle.load(f)
            model.eval()
            return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=-1)
        except Exception:
            os.remove(MODEL_CACHE)

    if progress_callback:
        progress_callback("首次加载模型（仅需一次）…")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)
    model.eval()

    with open(MODEL_CACHE, "wb") as f:
        pickle.dump((tokenizer, model), f)

    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=-1)


def analyze_sentiment(users, comments, sentiment_pipeline, progress_callback=None):
    data = []
    for idx, (user, c) in enumerate(zip(users, comments), start=1):
        try:
            result = sentiment_pipeline(c[:512])[0]
            label, score = result['label'], result['score']
            threshold = 0.6
            if score < threshold:
                sentiment = "中性"
            elif label == 'Positive':
                sentiment = "积极"
            elif label == 'Negative':
                sentiment = "消极"
            else:
                sentiment = "中性"
        except Exception:
            sentiment, score = "中性", 0
        data.append({"用户": user, "评论": c, "情绪": sentiment, "置信度": score})
        if progress_callback:
            progress_callback(f"分析评论 {idx}/{len(comments)}")

    return pd.DataFrame(data)


def visualize_sentiment(df, canvas_frame):
    summary = df["情绪"].value_counts(normalize=True) * 100
    fig, ax = plt.subplots(figsize=(5, 3))
    summary.plot(kind="bar", ax=ax, color=["green", "red", "gray"])
    ax.set_title("情绪分布", fontproperties="Microsoft YaHei")
    ax.set_ylabel("百分比", fontproperties="Microsoft YaHei")

    for i, v in enumerate(summary):
        ax.text(i, v + 0.5, f"{v:.1f}%", ha='center', fontname='Microsoft YaHei')

    plt.tight_layout()
    canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill='both', expand=True)


class App:
    def __init__(self, root):
        self.root = root
        root.title("微博评论情绪分析（Playwright异步版）")
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

        self.sentiment_pipeline = None
        threading.Thread(target=self.preload_model, daemon=True).start()

    def update_progress(self, msg):
        self.progress_label.config(text=msg)
        self.root.update()

    def preload_model(self):
        self.sentiment_pipeline = load_model(progress_callback=self.update_progress)
        self.update_progress("模型加载完成✅️")

    def start_analysis(self):
        weibo_url = self.entry.get().strip()
        if not weibo_url:
            messagebox.showwarning("提示", "请输入微博帖子链接")
            return
        if self.sentiment_pipeline is None:
            messagebox.showinfo("提示", "模型正在加载，请稍后再试…")
            return

        def run_async():
            asyncio.run(self._async_main(weibo_url))
        threading.Thread(target=run_async, daemon=True).start()

    async def _async_main(self, weibo_url):
        users, comments = await async_browser_workflow(weibo_url, self.update_progress)
        if len(comments) == 0:
            messagebox.showerror("错误", "未抓取到评论，请检查微博是否公开或评论区是否加载。")
            return

        self.update_progress("开始情绪分析…")
        df = analyze_sentiment(users, comments, self.sentiment_pipeline, progress_callback=self.update_progress)
        df.to_csv("weibo_comments_sentiment.csv", index=False, encoding="utf-8-sig")

        self.update_progress("分析完成，已生成 CSV 文件")
        visualize_sentiment(df, self.canvas_frame)


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
