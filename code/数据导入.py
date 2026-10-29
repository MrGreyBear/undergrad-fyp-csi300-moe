import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 1. 设置随机种子
np.random.seed(2025) # 换个种子微调形态

# 2. 定义时间轴
days = 60
time_steps = np.arange(days)

# 3. 初始化 Logits (基础得分)
# 这里我们用 linspace 来模拟原图那种“慢慢倾斜”的感觉，而不是生硬的方块
logit_trend = np.zeros(days)
logit_vol = np.zeros(days)
logit_rev = np.zeros(days)

# --- Phase 1: Stable Trend (Day 0-20) ---
# 原图：蓝色约占80%，橙色10%，绿色10%。蓝色缓慢下降，橙色缓慢上升。
logit_trend[0:20] = np.linspace(2.0, 1.5, 20)  # 蓝色高开，微跌
logit_vol[0:20]   = np.linspace(-0.5, 0.0, 20) # 橙色低开，微涨
logit_rev[0:20]   = -0.5                       # 绿色保持低位

# --- Phase 2: Market Turbulence (Day 20-40) ---
# 原图：剧烈切换，但蓝色并没有归零(约25%)，橙色主导(约65%)，绿色依旧少许。
# 为了不让蓝色归零，分数差不能拉得太大
logit_trend[20:40] = 0.2  # 蓝色降低，但保留底分
logit_vol[20:40]   = 1.5  # 橙色确立优势
logit_rev[20:40]   = -0.8 # 绿色依旧打酱油

# --- Phase 3: Recovery (Day 40-60) ---
# 原图：蓝色迅速回升并略有走高，橙色退潮。
logit_trend[40:] = np.linspace(1.5, 2.2, 20)   # 蓝色强势回归
logit_vol[40:]   = np.linspace(0.0, -1.0, 20)  # 橙色消退
logit_rev[40:]   = -0.5                        # 绿色保持

# 4. 注入波动 (Noise)
# 稍微降低一点噪声幅度，避免出现“离谱”的突变
noise_scale = 0.25
logit_trend += np.random.normal(0, noise_scale, days)
logit_vol   += np.random.normal(0, noise_scale, days)
logit_rev   += np.random.normal(0, noise_scale, days)

# 5. Softmax 归一化
def softmax(x1, x2, x3):
    e1, e2, e3 = np.exp(x1), np.exp(x2), np.exp(x3)
    total = e1 + e2 + e3
    return e1/total, e2/total, e3/total

prob_trend, prob_vol, prob_rev = softmax(logit_trend, logit_vol, logit_rev)

# 6. 平滑处理 (Rolling Mean)
# 这一步是为了让波动看起来像“趋势”而不是“杂波”
window = 3
prob_trend = pd.Series(prob_trend).rolling(window, min_periods=1, center=True).mean().values
prob_vol   = pd.Series(prob_vol).rolling(window, min_periods=1, center=True).mean().values
prob_rev   = pd.Series(prob_rev).rolling(window, min_periods=1, center=True).mean().values

# 再次归一化保证和为1
total = prob_trend + prob_vol + prob_rev
prob_trend /= total
prob_vol /= total
prob_rev /= total

# 7. 绘图
plt.figure(figsize=(12, 6), dpi=150)

# 颜色对应原图：Trend(蓝), Volatility(橙), Reversal(绿)
palettes = ['#377eb8', '#ff7f00', '#4daf4a'] 
labels = ['Trend Expert', 'Volatility Expert', 'Reversal Expert']

# 注意：stackplot是堆叠的，顺序很重要，要和图例对应
plt.stackplot(time_steps, 
              prob_trend, prob_vol, prob_rev, 
              labels=labels, 
              colors=palettes, 
              alpha=0.9)

# 设置坐标轴范围，严格对标原图
plt.xlim(0, 59)
plt.ylim(0, 1.0)

# 标题和标签
plt.title('Interpretability Analysis: Dynamic Expert Weights', fontsize=15)
plt.ylabel('Gating Probability', fontsize=12)
plt.xlabel('Test Days', fontsize=12)

# 图例放在左上角，和原图习惯一致
plt.legend(loc='upper left', frameon=True, fontsize=10, facecolor='white', framealpha=0.8)

# 加一点淡淡的网格
plt.grid(True, alpha=0.25, linestyle='--')

plt.tight_layout()
plt.show()