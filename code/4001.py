# -*- coding: utf-8 -*-
"""
Chapter 4.4: Model Residual Diagnostics (Fixed Version)
-------------------------------------------------------
针对数据集: "沪深300指数历史数据 12月1号.csv"
修复内容: 
    1. 修正 jarque_bera 返回值解包错误 (ValueError)。
    2. 独立运行，不依赖其他模块。

Output: 
    - Figure_4_4_Residuals.png (图表)
    - Chapter4_Residual_Metrics.csv (数据表)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import scipy.stats as stats
import os
import warnings

# 统计检验库
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

# 设置绘图风格 (支持中文显示)
# 尝试多种常见中文字体，防止乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')
warnings.filterwarnings('ignore')

# 设置随机种子
def set_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seeds(42)

# ==============================================================================
# 1. 专用数据清洗 (针对 "沪深300指数历史数据 12月1号.csv")
# ==============================================================================
def load_specific_data(filename):
    print(f"正在加载数据: {filename} ...")
    
    # 自动查找文件 (防止路径错误)
    if not os.path.exists(filename):
        print(f"提示: 当前目录下未找到 {filename}")
        possible_files = [f for f in os.listdir('.') if f.endswith('.csv')]
        print(f"当前目录下的CSV文件: {possible_files}")
        # 尝试匹配最像的文件
        for f in possible_files:
            if "沪深300" in f or "csi300" in f.lower():
                filename = f
                print(f"自动选择文件: {filename}")
                break
        else:
            return None

    # 读取 CSV (尝试常见编码)
    try:
        df = pd.read_csv(filename, encoding='utf-8')
    except:
        try:
            df = pd.read_csv(filename, encoding='gbk')
        except:
            print("错误: 无法读取文件，请检查编码。")
            return None

    # 1. 重命名列 (中文 -> 英文)
    # 假设列名是: "日期", "收盘", "开盘", "高", "低", "交易量", "涨跌幅"
    rename_map = {
        '日期': 'Date', 
        '收盘': 'Close', 
        '开盘': 'Open', 
        '高': 'High', 
        '低': 'Low'
    }
    df.rename(columns=rename_map, inplace=True)
    
    # 2. 数据清洗: 去除千分位逗号 (例如 "3,840.23" -> 3840.23)
    cols_to_fix = ['Close', 'Open', 'High', 'Low']
    for col in cols_to_fix:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.replace(',', '').astype(float)

    # 3. 时间处理
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True) # 确保按时间正序排列
    else:
        print("错误: 未找到'日期'或'Date'列")
        return None

    # 4. 计算对数收益率 (Log Return)
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # 5. 设置预测目标 (Target): 预测下一天的收益率
    df['Target'] = df['Log_Ret'].shift(-1)
    
    df.dropna(inplace=True)
    
    print(f"数据加载完成。样本数: {len(df)}")
    return df

# ==============================================================================
# 2. 模型训练与残差获取
# ==============================================================================
def get_residuals(df):
    # 准备数据
    data = df[['Log_Ret']].values
    targets = df['Target'].values
    
    # 分割训练集和测试集 (最后 60 天用于残差分析)
    test_len = 60
    if len(data) < 200: test_len = int(len(data) * 0.2)
        
    train_data = data[:-test_len]
    train_targets = targets[:-test_len]
    test_data = data[-test_len:]
    test_targets = targets[-test_len:]
    
    # 归一化
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)
    
    # 创建序列数据 (Sequence)
    seq_len = 10
    def make_seq(x, y):
        X_seq, y_seq = [], []
        for i in range(len(x) - seq_len):
            X_seq.append(x[i:i+seq_len])
            y_seq.append(y[i+seq_len])
        return np.array(X_seq), np.array(y_seq)
    
    X_train, y_train = make_seq(train_scaled, train_targets)
    X_test, y_test = make_seq(test_scaled, test_targets)
    
    if len(X_train) == 0 or len(X_test) == 0:
        print("错误: 数据量太少，无法构建序列。")
        return None

    # 建立一个简单的 LSTM 模型来拟合数据
    print("\n正在训练基础模型以生成残差...")
    model = Sequential([
        LSTM(32, input_shape=(seq_len, 1), return_sequences=False),
        Dense(1)
    ])
    model.compile(optimizer=Adam(0.001), loss='mse')
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
    
    # 预测并计算残差
    y_pred = model.predict(X_test, verbose=0).flatten()
    
    # 残差 = 真实值 - 预测值
    # 注意: y_test 此时是归一化后的 Target (收益率)。
    # 既然我们要分析残差的分布特性(正态性等)，直接用归一化后的残差也是可以的(统计性质不变)。
    # 或者反归一化回去，这里为了简便直接用 Scaled Residuals，不影响 JB/ADF 检验结果。
    residuals = y_test - y_pred
    
    return residuals

# ==============================================================================
# 3. 统计检验 (替代 QQ 图) - 已修复
# ==============================================================================
def run_statistical_tests(residuals):
    print("\n" + "="*60)
    print("RESIDUAL DIAGNOSTICS (统计检验报告)")
    print("="*60)
    
    results = {}
    
    # 1. 基础描述统计
    mu = np.mean(residuals)
    sigma = np.std(residuals)
    skew = stats.skew(residuals)
    kurt = stats.kurtosis(residuals) # Excess Kurtosis
    
    results['Mean'] = mu
    results['Std_Dev'] = sigma
    results['Skewness'] = skew
    results['Kurtosis'] = kurt
    
    print(f"Mean (均值):       {mu:.6f} (应接近 0)")
    print(f"Skewness (偏度):   {skew:.4f} (正态分布为 0)")
    print(f"Kurtosis (峰度):   {kurt:.4f} (正态分布为 0, >0 为尖峰)")
    
    # 2. 正态性检验: Jarque-Bera
    # *** 修复: 接收 4 个返回值 ***
    try:
        jb_stat, jb_p, jb_skew, jb_kurt = jarque_bera(residuals)
    except ValueError:
        # 万一版本极旧
        jb_stat, jb_p = jarque_bera(residuals)
        
    results['JB_Stat'] = jb_stat
    results['JB_P_Value'] = jb_p
    print("-" * 40)
    print(f"Jarque-Bera Test: Stat={jb_stat:.2f}, P-value={jb_p:.4f}")
    if jb_p < 0.05:
        print(">> 结论: 拒绝原假设。残差不服从完美正态分布 (Acceptable in Finance)。")
    else:
        print(">> 结论: 接受原假设。残差服从正态分布。")
        
    # 3. 平稳性检验: ADF Test
    adf_res = adfuller(residuals)
    results['ADF_Stat'] = adf_res[0]
    results['ADF_P_Value'] = adf_res[1]
    print("-" * 40)
    print(f"ADF Test:         Stat={adf_res[0]:.2f}, P-value={adf_res[1]:.4f}")
    if adf_res[1] < 0.05:
        print(">> 结论: 拒绝原假设。残差是平稳的 (Stationary)。模型提取了趋势。")
    else:
        print(">> 结论: 接受原假设。残差可能不平稳。")
        
    # 4. 自相关性检验: Ljung-Box Test (白噪声检验)
    lb_res = acorr_ljungbox(residuals, lags=[10], return_df=True)
    lb_stat = lb_res.iloc[0]['lb_stat']
    lb_p = lb_res.iloc[0]['lb_pvalue']
    results['LB_Stat'] = lb_stat
    results['LB_P_Value'] = lb_p
    print("-" * 40)
    print(f"Ljung-Box (Lag10): Stat={lb_stat:.2f}, P-value={lb_p:.4f}")
    if lb_p > 0.05:
        print(">> 结论: 接受原假设。无显著自相关 (白噪声)，模型信息提取充分。")
    else:
        print(">> 结论: 拒绝原假设。残差仍存在自相关。")
        
    # 5. Durbin-Watson (一阶自相关)
    dw = durbin_watson(residuals)
    results['Durbin_Watson'] = dw
    print(f"Durbin-Watson:    {dw:.2f} (接近 2.0 表示无自相关)")
    
    print("="*60)
    return results

# ==============================================================================
# 4. 绘图 (直方图替代 QQ 图)
# ==============================================================================
def plot_residuals(residuals):
    plt.figure(figsize=(12, 5))
    
    # 左图: 残差时间序列 (检查平稳性)
    plt.subplot(1, 2, 1)
    plt.plot(residuals, color='#1f77b4', linewidth=1)
    plt.axhline(0, color='red', linestyle='--', linewidth=1.5)
    plt.title('Residuals over Time (Time Series)', fontsize=12)
    plt.xlabel('Time Step')
    plt.ylabel('Error Value')
    plt.grid(True, alpha=0.3)
    
    # 右图: 残差分布直方图 (检查正态性) - 替代 QQ 图
    plt.subplot(1, 2, 2)
    sns.histplot(residuals, kde=True, color='green', stat='density', bins=30, label='Residuals')
    
    # 叠加标准正态分布曲线
    mu, std = stats.norm.fit(residuals)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)
    plt.plot(x, p, 'r--', linewidth=2, label='Normal Dist.')
    
    plt.title('Residual Distribution (Histogram)', fontsize=12)
    plt.xlabel('Residual Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    # 保存文件
    plt.savefig('Figure_4_4_Residuals.png', dpi=300)
    print("\n图表已保存至: Figure_4_4_Residuals.png")
    plt.show()

# ==============================================================================
# 主程序
# ==============================================================================
if __name__ == "__main__":
    # 指定特定文件名
    filename = '沪深300指数历史数据 12月1号.csv'
    
    # 1. 加载数据
    df = load_specific_data(filename)
    
    if df is not None:
        # 2. 获取残差
        residuals = get_residuals(df)
        
        if residuals is not None:
            # 3. 统计检验
            metrics = run_statistical_tests(residuals)
            
            # 4. 绘图
            plot_residuals(residuals)
            
            # 5. 保存结果表
            pd.DataFrame([metrics]).to_csv('Chapter4_Residual_Metrics.csv', index=False)
            print("统计指标已保存至: Chapter4_Residual_Metrics.csv")