# -*- coding: utf-8 -*-
"""
Robustness Analysis & Rolling Cross-Validation (Long-Short Strategy)
--------------------------------------------------------------------
Settings: 
    - Train Window: 756 days (3 years)
    - Test Window:  63 days  (1 quarter)
    - Step Size:    63 days
    - Data:         2018 to Present (2.csi300_akshare_data.csv)

Description: 
    [Updated Feature]: 
    计算并展示每个滚动测试窗口(Fold)的独立收益率 (Period Return)。
    包含正则化 (L2, BN, Dropout) 以保持高胜率。

Author: Assistant (Based on User's FYP Architecture)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import os
import random
import warnings
import gc

# 使用 StandardScaler 替代 MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Conv1D, GlobalAveragePooling1D, Concatenate, Layer, BatchNormalization, Lambda, Multiply, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Set seeds for reproducibility
def set_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seeds(42)

# ==============================================================================
# 1. Data Loading & Feature Engineering
# ==============================================================================
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(series, fast=12, slow=26, signal=9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd - signal_line

def calculate_atr(high, low, close, window=14):
    tr1 = high - low
    tr2 = np.abs(high - close.shift(1))
    tr3 = np.abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=window).mean()

def load_and_process_data(filepath):
    print(f"Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: {filepath} not found.")
        return None
    
    # Standardize column names
    df.columns = [c.capitalize() for c in df.columns]
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    
    # Check required columns
    req_cols = ['Open', 'High', 'Low', 'Close']
    for c in req_cols:
        if c not in df.columns:
            raise ValueError(f"Data must contain '{c}' column.")
        
    # --- Feature Engineering ---
    # Target: Log Returns
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Features
    df['Vol_20'] = df['Log_Ret'].rolling(window=20).std()
    df['ATR_14'] = calculate_atr(df['High'], df['Low'], df['Close'])
    df['MA_5_Dist'] = df['Close'] / df['Close'].rolling(window=5).mean() - 1
    df['MA_20_Dist'] = df['Close'] / df['Close'].rolling(window=20).mean() - 1
    df['MACD'] = calculate_macd(df['Close'])
    df['RSI_14'] = calculate_rsi(df['Close'])
    df['Mom_10'] = df['Close'] / df['Close'].shift(10) - 1
    
    df.dropna(inplace=True)
    df.replace([np.inf, -np.inf], 0, inplace=True)
    
    print(f"Data processed. Shape: {df.shape}")
    return df

def create_sequences(data, seq_length, target_col_idx):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length, target_col_idx])
    return np.array(X), np.array(y)

# ==============================================================================
# 2. Model Architecture (Regularized MoE)
# ==============================================================================
def build_moe_model(input_shape):
    inputs = Input(shape=input_shape, name='Input_Layer')
    reg_val = 1e-4 
    
    # Expert 1: LSTM
    x1 = LSTM(64, return_sequences=True, kernel_regularizer=l2(reg_val))(inputs)
    x1 = BatchNormalization()(x1)
    x1 = Dropout(0.3)(x1)
    x1 = LSTM(32, return_sequences=False, kernel_regularizer=l2(reg_val))(x1)
    x1 = BatchNormalization()(x1)
    x1 = Dropout(0.3)(x1)
    out1 = Dense(1, name='Expert_LSTM_Out', kernel_regularizer=l2(reg_val))(x1)
    
    # Expert 2: CNN
    x2 = Conv1D(filters=32, kernel_size=3, padding='same', kernel_regularizer=l2(reg_val))(inputs)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    x2 = GlobalAveragePooling1D()(x2)
    x2 = Dropout(0.3)(x2)
    x2 = Dense(16, activation='relu', kernel_regularizer=l2(reg_val))(x2)
    out2 = Dense(1, name='Expert_CNN_Out', kernel_regularizer=l2(reg_val))(x2)
    
    # Expert 3: MLP
    x3 = tf.keras.layers.Flatten()(inputs)
    x3 = Dense(64, kernel_regularizer=l2(reg_val))(x3)
    x3 = BatchNormalization()(x3)
    x3 = Activation('relu')(x3)
    x3 = Dropout(0.3)(x3)
    out3 = Dense(1, name='Expert_MLP_Out', kernel_regularizer=l2(reg_val))(x3)
    
    # Gating
    gate_hidden = tf.keras.layers.Flatten()(inputs)
    gate_hidden = Dense(32, kernel_regularizer=l2(reg_val))(gate_hidden)
    gate_hidden = BatchNormalization()(gate_hidden)
    gate_hidden = Activation('relu')(gate_hidden)
    gate_hidden = Dropout(0.2)(gate_hidden)
    gate_weights = Dense(3, activation='softmax', name='Gating_Weights', kernel_regularizer=l2(reg_val))(gate_hidden)
    
    # Output
    experts_stack = Lambda(lambda x: tf.stack(x, axis=1), name='Stack_Experts')([out1, out2, out3])
    gate_weights_expanded = Lambda(lambda x: tf.expand_dims(x, axis=-1), name='Expand_Gate')(gate_weights)
    weighted_experts = Multiply(name='Weighted_Experts')([experts_stack, gate_weights_expanded])
    final_output = Lambda(lambda x: tf.reduce_sum(x, axis=1), name='Final_Output')(weighted_experts)
    
    model = Model(inputs=inputs, outputs=final_output)
    opt = Adam(learning_rate=0.0005, clipnorm=1.0)
    model.compile(optimizer=opt, loss='mse', metrics=['mae'])
    
    return model

# ==============================================================================
# 3. Rolling Window Validation Engine
# ==============================================================================
def run_rolling_analysis(df, train_window=756, test_window=63, step_size=63, seq_length=10):
    
    # Define Feature List
    target_col = 'Log_Ret'
    feature_cols = ['Log_Ret', 'Vol_20', 'ATR_14', 'MA_5_Dist', 'MA_20_Dist', 'MACD', 'RSI_14', 'Mom_10']
    
    # Check features
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    data_values = df[feature_cols].values
    
    results = []
    predictions_list = []
    
    total_samples = len(data_values)
    start_index = 0
    
    print(f"\nStarting Rolling Validation (MoE Model)...")
    print(f"Total Data Points: {total_samples}")
    print(f"Train: {train_window}, Test: {test_window}, Step: {step_size}")
    print("-" * 75)
    print(f"{'Fold':<6} | {'Date Range':<22} | {'RMSE':<8} | {'Acc':<8} | {'Return':<8}")
    print("-" * 75)
    
    iteration = 1
    
    while start_index + train_window + test_window + seq_length < total_samples:
        train_start = start_index
        train_end = start_index + train_window
        test_start = train_end
        test_end = train_end + test_window
        
        # 1. Slice
        train_raw = data_values[train_start:train_end]
        test_raw = data_values[test_start:test_end]
        
        # 2. Scale
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_raw)
        test_scaled = scaler.transform(test_raw)
        
        # 3. Sequences
        full_test_context = np.concatenate((train_scaled[-seq_length:], test_scaled))
        X_train, y_train = create_sequences(train_scaled, seq_length, 0)
        X_test, y_test = create_sequences(full_test_context, seq_length, 0)
        
        if len(X_train) == 0 or len(X_test) == 0:
            break

        # 4. Train
        model = build_moe_model((seq_length, len(feature_cols)))
        early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=4, min_lr=0.00001)
        
        model.fit(X_train, y_train, epochs=40, batch_size=32, verbose=0, callbacks=[early_stop, reduce_lr])
        
        # 5. Predict
        preds_scaled = model.predict(X_test, verbose=0)
        
        # 6. Inverse
        dummy_pred = np.zeros((len(preds_scaled), len(feature_cols)))
        dummy_pred[:, 0] = preds_scaled.flatten()
        preds_inv = scaler.inverse_transform(dummy_pred)[:, 0]
        
        dummy_actual = np.zeros((len(y_test), len(feature_cols)))
        dummy_actual[:, 0] = y_test.flatten()
        actual_inv = scaler.inverse_transform(dummy_actual)[:, 0]
        
        # 7. Metrics
        mse = mean_squared_error(actual_inv, preds_inv)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual_inv, preds_inv)
        
        # Directional Accuracy
        direction_match = np.sign(preds_inv) == np.sign(actual_inv)
        acc = np.mean(direction_match)
        
        # --- NEW: Calculate Period Return (Long-Short) ---
        # 逻辑: 预测收益率符号 * 实际收益率
        # 如果预测 > 0 (做多)，得到实际收益 (涨赚跌亏)
        # 如果预测 < 0 (做空)，得到反向收益 (跌赚涨亏)
        daily_strat_ret = np.sign(preds_inv) * actual_inv
        
        # 累计该阶段收益 (Log return summation, then exp to get simple return)
        period_log_return = np.sum(daily_strat_ret)
        period_return = np.exp(period_log_return) - 1
        
        fold_dates = df.index[test_start:test_start+len(y_test)]
        date_str = f"{fold_dates[0].date()}~{fold_dates[-1].date()}"
        
        results.append({
            'Fold': iteration,
            'Date_Start': fold_dates[0],
            'Date_End': fold_dates[-1],
            'RMSE': rmse,
            'MAE': mae,
            'Direction_Acc': acc,
            'Period_Return': period_return # New Metric
        })
        
        for date, pred, act in zip(fold_dates, preds_inv, actual_inv):
            predictions_list.append({'Date': date, 'Pred': pred, 'Actual': act})
            
        print(f"{iteration:<6} | {date_str:<22} | {rmse:<8.5f} | {acc:<8.2%} | {period_return:<8.2%}")
        
        # Cleanup
        tf.keras.backend.clear_session()
        del model
        gc.collect()

        start_index += step_size
        iteration += 1

    return pd.DataFrame(results), pd.DataFrame(predictions_list)

# ==============================================================================
# 4. Visualization & Reporting
# ==============================================================================
def plot_robustness_detailed(metrics_df, preds_df):
    plt.style.use('seaborn-v0_8-whitegrid')
    
    preds_df = preds_df.set_index('Date').sort_index()
    
    # Strategy Calculation
    preds_df['Ret_LongOnly'] = np.where(preds_df['Pred'] > 0, preds_df['Actual'], 0)
    preds_df['Ret_LongShort'] = np.sign(preds_df['Pred']) * preds_df['Actual']
    
    preds_df['Cum_Benchmark'] = (1 + preds_df['Actual']).cumprod()
    preds_df['Cum_LongOnly'] = (1 + preds_df['Ret_LongOnly']).cumprod()
    preds_df['Cum_LongShort'] = (1 + preds_df['Ret_LongShort']).cumprod()
    
    # Figure 1: Cumulative Wealth
    plt.figure(figsize=(12, 6))
    plt.plot(preds_df.index, preds_df['Cum_Benchmark'], label='Benchmark (CSI 300)', color='gray', linestyle='--', alpha=0.6)
    plt.plot(preds_df.index, preds_df['Cum_LongOnly'], label='Long-Only Strategy', color='royalblue', linewidth=1.5, alpha=0.8)
    plt.plot(preds_df.index, preds_df['Cum_LongShort'], label='Long-Short MoE Strategy', color='#d62728', linewidth=2.5)
    
    plt.title('Performance Comparison: Train=756, Test=63', fontsize=16)
    plt.ylabel('Cumulative Wealth (Start = 1.0)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    final_ls = preds_df['Cum_LongShort'].iloc[-1]
    print(f"\nFinal Long-Short Total Return: {(final_ls - 1)*100:.2f}%")

    # Figure 2: Metrics Boxplot & Distribution
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    sns.boxplot(y=metrics_df['RMSE'], color='skyblue')
    plt.title('Robustness: RMSE Distribution', fontsize=14)
    
    plt.subplot(1, 2, 2)
    sns.histplot(metrics_df['Direction_Acc'], kde=True, color='orange', bins=10)
    plt.title('Robustness: Directional Accuracy', fontsize=14)
    plt.axvline(0.5, color='red', linestyle='--')
    plt.tight_layout()
    plt.show()
    
    # --- NEW Figure 3: Per-Fold Return Bar Chart ---
    plt.figure(figsize=(12, 5))
    
    # Color logic: Red for profit, Green for loss (Chinese style) or Green/Red (US style)
    # Let's use Red (>0) and Green (<0) which is standard in many quantitative tools
    colors = ['#d62728' if x >= 0 else 'green' for x in metrics_df['Period_Return']]
    
    plt.bar(metrics_df['Fold'], metrics_df['Period_Return'], color=colors, alpha=0.8)
    plt.axhline(0, color='black', linewidth=1)
    plt.title('Return per Rolling Fold (Quarterly Performance)', fontsize=16)
    plt.xlabel('Fold Number (Time Sequence)', fontsize=12)
    plt.ylabel('Return Rate', fontsize=12)
    
    # Format y-axis as percentage
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()

# ==============================================================================
# Main Execution
# ==============================================================================
if __name__ == "__main__":
    data_file = '2.csi300_akshare_data.csv'
    
    if not os.path.exists(data_file):
        print(f"Error: {data_file} not found. Please upload it.")
    else:
        df = load_and_process_data(data_file)
        
        if df is not None:
            metrics, predictions = run_rolling_analysis(
                df, 
                train_window=756, 
                test_window=63, 
                step_size=63
            )
            
            print("\n=== Robustness Analysis Summary ===")
            print(metrics[['Fold', 'Date_Start', 'RMSE', 'Direction_Acc', 'Period_Return']].to_string(index=False))
            
            plot_robustness_detailed(metrics, predictions)
            
            metrics.to_csv('robustness_metrics_returns.csv', index=False)
            predictions.to_csv('robustness_predictions_returns.csv')
            print("\nResults saved.")