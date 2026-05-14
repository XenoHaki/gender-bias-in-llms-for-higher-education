import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from matplotlib import font_manager
import pandas as pd
import numpy as np

BASE = Path(__file__).resolve().parents[1]
OUTDIR = BASE / "analysis" / "figures" / "paper_figures"
OUTDIR.mkdir(parents=True, exist_ok=True)


def configure_fonts():
    candidates = [
        "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC",
        "WenQuanYi Zen Hei", "Arial Unicode MS", "PingFang SC"
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 140
    plt.rcParams["savefig.dpi"] = 300


configure_fonts()

RED_ORANGE = ["#b2182b", "#d6604d", "#f4a261", "#f6bd60", "#ffd6a5"]
BLUE_GRAD = ["#0b3c5d", "#1d5f8c", "#3f88c5", "#73a9d8", "#b9d6ea"]
GREEN_GRAD = ["#1b5e20", "#2e7d32", "#4caf50", "#81c784", "#c8e6c9"]
MALE = "#4C78A8"
FEMALE = "#F58518"
EN_COLOR = "#4C78A8"
ZH_COLOR = "#E45756"
HILITE = "#E15759"
BASEBAR = "#76B7B2"
FEATURE_LABELS = {
    "model": "模型",
    "major": "专业",
    "gpa": "绩点",
    "competition": "科研竞赛",
    "internship": "实习经历",
    "english": "英语水平",
    "gender": "性别",
    "language": "语言",
    "measurement": "测量轮次",
}


def gradient(palette, n):
    if n <= len(palette):
        return palette[:n]
    steps = np.linspace(0, len(palette) - 1, n)
    idx = np.floor(steps).astype(int)
    frac = steps - idx
    colors = []
    for i, f in zip(idx, frac):
        if i >= len(palette) - 1:
            colors.append(palette[-1])
            continue
        c1 = np.array(matplotlib.colors.to_rgb(palette[i]))
        c2 = np.array(matplotlib.colors.to_rgb(palette[i + 1]))
        colors.append(to_hex(c1 * (1 - f) + c2 * f))
    return colors


def save(fig, name):
    png = OUTDIR / f"{name}.png"
    fig.tight_layout()
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)
    print(png)


# Figure 1: logOR counts
with open(BASE / r"analysis\text_bias\logor_summary.json", "r", encoding="utf-8") as f:
    raw = json.load(f)
logor = pd.DataFrame(raw)
logor = logor[(logor["task"] == "downstream") & (logor["model"] == "__ALL_MODELS__")].copy()
logor["language_label"] = logor["language"].map({"en": "英文", "zh": "中文"})
metrics = [
    ("male_biased", "男性偏向词", BLUE_GRAD[1]),
    ("female_biased", "女性偏向词", RED_ORANGE[2]),
    ("neutral", "中性词", GREEN_GRAD[2]),
]
fig, ax = plt.subplots(figsize=(7, 4.5))
x = np.arange(len(logor))
width = 0.22
for i, (metric, label, color) in enumerate(metrics):
    vals = logor[metric].to_numpy()
    ax.bar(x + (i - 1) * width, vals, width=width, label=label, color=color)
ax.set_xticks(x)
ax.set_xticklabels(logor["language_label"])
ax.set_ylabel("词项数量")
ax.set_title("中英文 OR 词汇偏向数量对比")
ax.legend(frameon=False, ncol=3, loc="upper center")
ax.spines[["top", "right"]].set_visible(False)
save(fig, "fig1_logor_counts")


# Figure 2: WEAT comparison
order = [
    "gender_names", "leadership_vs_support", "career_vs_family", "competence_vs_warmth",
    "agency_vs_communion", "stem_vs_care", "rationality_vs_emotionality",
    "power_vs_dependence", "risk_vs_caution", "public_vs_domestic_roles"
]
label_map = {
    "gender_names": "性别姓名",
    "leadership_vs_support": "领导 vs 支持",
    "career_vs_family": "事业 vs 家庭",
    "competence_vs_warmth": "能力 vs 温暖",
    "agency_vs_communion": "主体性 vs 共融性",
    "stem_vs_care": "技术 vs 照护",
    "rationality_vs_emotionality": "理性 vs 感性",
    "power_vs_dependence": "权力 vs 依赖",
    "risk_vs_caution": "冒险 vs 谨慎",
    "public_vs_domestic_roles": "公共角色 vs 家庭角色",
}
en_df = pd.read_csv(BASE / r"analysis\text_bias\weat_downstream_only\weat_summary.csv")[["attribute_set", "effect_size_cohen_d"]].rename(columns={"effect_size_cohen_d": "EN"})
zh_df = pd.read_csv(BASE / r"analysis\text_bias\weat_downstream_zh\weat_summary.csv")[["attribute_set", "effect_size_cohen_d"]].rename(columns={"effect_size_cohen_d": "ZH"})
weat = pd.DataFrame({"attribute_set": order}).merge(en_df, on="attribute_set", how="left").merge(zh_df, on="attribute_set", how="left")
weat["label"] = weat["attribute_set"].map(label_map)
fig, ax = plt.subplots(figsize=(9.5, 6.2))
y = np.arange(len(weat))
height = 0.36
bars1 = ax.barh(y - height/2, weat["EN"].to_numpy(), height=height, color=EN_COLOR, label="英文")
bars2 = ax.barh(y + height/2, weat["ZH"].to_numpy(), height=height, color=ZH_COLOR, label="中文")
ax.axvline(0, color="#444444", linewidth=1)
ax.set_yticks(y)
ax.set_yticklabels(weat["label"])
ax.invert_yaxis()
ax.set_xlabel("Cohen's d")
ax.set_title("中英文 WEAT 效应量对照")
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
save(fig, "fig2_weat_comparison")


# Figure 3: sentiment by discipline and gender
stats = pd.read_csv(BASE / r"analysis\text_bias\sentiment\sentiment_group_stats.csv")
tests = pd.read_csv(BASE / r"analysis\text_bias\sentiment\sentiment_tests.csv")
stats = stats[(stats["task"] == "downstream") & (stats["model"] == "__ALL_MODELS__") & (stats["discipline"] != "__ALL_DISCIPLINES__")].copy()
tests = tests[(tests["task"] == "downstream") & (tests["model"] == "__ALL_MODELS__") & (tests["discipline"] != "__ALL_DISCIPLINES__")].copy()
order_en = ["Chinese Language and Literature", "Computer Science and Technology", "Mathematics and Applied Mathematics", "Sociology"]
order_zh = ["\u6c49\u8bed\u8a00\u6587\u5b66", "\u8ba1\u7b97\u673a\u79d1\u5b66\u4e0e\u6280\u672f", "\u6570\u5b66\u4e0e\u5e94\u7528\u6570\u5b66", "\u793e\u4f1a\u5b66"]
short_en = {
    "Chinese Language and Literature": "汉语言文学",
    "Computer Science and Technology": "计算机",
    "Mathematics and Applied Mathematics": "数学",
    "Sociology": "社会学",
}
short_zh = {
    "\u6c49\u8bed\u8a00\u6587\u5b66": "\u6c49\u8bed\u8a00\u6587\u5b66",
    "\u8ba1\u7b97\u673a\u79d1\u5b66\u4e0e\u6280\u672f": "\u8ba1\u7b97\u673a",
    "\u6570\u5b66\u4e0e\u5e94\u7528\u6570\u5b66": "\u6570\u5b66",
    "\u793e\u4f1a\u5b66": "\u793e\u4f1a\u5b66",
}
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=False)
for ax, lang, order_disc, short_map in zip(axes, ["en", "zh"], [order_en, order_zh], [short_en, short_zh]):
    sub = stats[stats["language"] == lang].copy()
    pivot = sub.pivot(index="discipline", columns="gender", values="mean_sentiment").reindex(order_disc)
    x = np.arange(len(pivot))
    w = 0.34
    male = pivot["male"].to_numpy()
    female = pivot["female"].to_numpy()
    bm = ax.bar(x - w/2, male, width=w, color=MALE, label="男性")
    bf = ax.bar(x + w/2, female, width=w, color=FEMALE, label="女性")
    ax.axhline(0, color="#444444", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([short_map[d] for d in pivot.index])
    ax.set_title("英文" if lang == "en" else "中文")
    if lang == "en":
        ax.set_ylim(0.17, 0.215)
    else:
        ax.set_ylim(-0.004, 0.018)
    ax.spines[["top", "right"]].set_visible(False)
    tsub = tests[tests["language"] == lang].set_index("discipline")
    for i, d in enumerate(pivot.index):
        if d in tsub.index and bool(tsub.loc[d, "t_significant_0_05"]):
            ymax = max(float(male[i]), float(female[i]))
            ymin = min(float(male[i]), float(female[i]))
            y0, y1 = ax.get_ylim()
            yr = y1 - y0
            offset = max(yr * 0.015, abs(ymax - ymin) * 0.15 if ymax != ymin else yr * 0.015)
            ystar = min(ymax + offset, y1 - yr * 0.06)
            ax.plot([i - w/2, i + w/2], [ystar, ystar], color="#333333", linewidth=1)
            ax.text(i, min(ystar + yr * 0.01, y1 - yr * 0.03), "*", ha="center", va="bottom", fontsize=12)
axes[0].set_ylabel("平均情感分数")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, ncol=1, loc="center right", bbox_to_anchor=(0.995, 0.67))
fig.suptitle("推荐信情感分数：语言、专业与性别分组", y=0.98)
save(fig, "fig3_sentiment_by_discipline")


# Figure 4: overall SHAP importance
lgb = pd.read_csv(BASE / r"analysis\modeling\lightgbm_resume\shap_summary.csv")
cat = pd.read_csv(BASE / r"analysis\modeling\catboost_resume\shap_summary.csv")
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
for ax, df, title, palette in zip(axes, [lgb, cat], ["LightGBM", "CatBoost"], [BLUE_GRAD, GREEN_GRAD]):
    top = df.sort_values("mean_abs_shap", ascending=True).tail(8).copy()
    top["feature_label"] = top["feature"].map(lambda x: FEATURE_LABELS.get(x, x))
    colors = list(reversed(gradient(palette, len(top))))
    ax.barh(top["feature_label"], top["mean_abs_shap"], color=colors)
    ax.set_title(title)
    ax.set_xlabel("Mean |SHAP|")
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("简历打分总体 SHAP 特征重要性", y=0.98)
save(fig, "fig4_shap_importance_overall")


# Figure 5: gender interaction SHAP
inter = pd.read_csv(BASE / r"analysis\modeling\catboost_resume_interaction\gender_interaction_shap_summary.csv")
feat_order = ["model", "major", "gpa", "competition", "internship", "english"]
inter = inter[inter["feature"].isin(feat_order)].copy()
inter["feature"] = pd.Categorical(inter["feature"], categories=feat_order, ordered=True)
inter = inter.sort_values("feature", ascending=True)
fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.barh([FEATURE_LABELS.get(f, f) for f in inter["feature"]], inter["mean_abs_interaction_with_gender"], color=gradient(RED_ORANGE, len(inter)))
ax.set_xlabel("与性别的平均绝对交互 SHAP")
ax.set_title("以性别为中心的交互 SHAP")
ax.spines[["top", "right"]].set_visible(False)
save(fig, "fig5_gender_interaction_shap")


# Figure 6a and 6b: by-major gender SHAP
major_dirs_en = {
    "Chinese Literature": BASE / r"analysis\modeling\catboost_resume_by_major_en\chinese-language-and-literature\shap_summary_by_gender.csv",
    "Computer Science": BASE / r"analysis\modeling\catboost_resume_by_major_en\computer-science-and-technology\shap_summary_by_gender.csv",
    "Mathematics": BASE / r"analysis\modeling\catboost_resume_by_major_en\mathematics-and-applied-mathematics\shap_summary_by_gender.csv",
    "Sociology": BASE / r"analysis\modeling\catboost_resume_by_major_en\sociology\shap_summary_by_gender.csv",
}
major_dirs_zh = {
    "?????": BASE / r"analysis\modeling\catboost_resume_by_major_zh\chinese-literature\shap_summary_by_gender.csv",
    "????????": BASE / r"analysis\modeling\catboost_resume_by_major_zh\computer-science\shap_summary_by_gender.csv",
    "???????": BASE / r"analysis\modeling\catboost_resume_by_major_zh\mathematics\shap_summary_by_gender.csv",
    "???": BASE / r"analysis\modeling\catboost_resume_by_major_zh\sociology\shap_summary_by_gender.csv",
}

def load_major_gender(paths):
    rows = []
    for major, path in paths.items():
        df = pd.read_csv(path)
        df = df[df["feature"] == "gender"][["gender", "mean_shap"]].copy()
        df["major"] = major
        rows.append(df)
    return pd.concat(rows, ignore_index=True)

for paths, fname, title, ticklabels in [
    (major_dirs_en, "fig6a_gender_shap_by_major_en", "gender mean SHAP by major (English)", ["汉语言文学", "计算机", "数学", "社会学"]),
    (major_dirs_zh, "fig6b_gender_shap_by_major_zh", "gender mean SHAP by major (Chinese)", ["汉语言文学", "计算机", "数学", "社会学"]),
]:
    df = load_major_gender(paths)
    order_maj = list(paths.keys())
    pivot = df.pivot(index="major", columns="gender", values="mean_shap").reindex(order_maj)
    x = np.arange(len(order_maj))
    w = 0.34
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    bm = ax.bar(x - w/2, pivot["male"], width=w, color=MALE, label="男性")
    bf = ax.bar(x + w/2, pivot["female"], width=w, color=FEMALE, label="女性")
    ax.axhline(0, color="#444444", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(ticklabels)
    ax.set_ylabel("性别 mean SHAP")
    ax.set_title("分专业性别 SHAP（英文）" if "English" in title else "分专业性别 SHAP（中文）")
    ax.legend(frameon=False, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, fname)

print("DONE")
