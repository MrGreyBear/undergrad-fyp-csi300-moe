# -*- coding: utf-8 -*-
"""
Improved MoE: Boost DA to ~63%+ (Higher penalty, larger model, more features)
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
from tensorflow.keras import backend as K
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Conv1D, GlobalAveragePooling1D, Concatenate, Layer, BatchNormalization, Lambda, Multiply, Activation, Attention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.ensemble import GradientBoostingRegressor
from tensorflow.keras import layers

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def set_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seeds(42)

# Feature Engineering (Added normalized Volume)
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

def calculate_bollinger(close, window=20):
    ma = close.rolling(window=window).mean()
    std = close.rolling(window=window).std()
    upper = ma + 2 * std
    lower = ma - 2 * std
    bb_pos = (close - lower) / (upper - lower)
    return bb_pos

def load_and_process_data(filepath):
    print(f"Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: {filepath} not found.")
        return None
    
    df.columns = [c.capitalize() for c in df.columns]
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    prediction_horizon = 3
    df['Target_Ret'] = df['Log_Ret'].rolling(window=prediction_horizon).mean().shift(-prediction_horizon)
    
    df['Vol_20'] = df['Log_Ret'].rolling(window=20).std()
    df['ATR_14'] = calculate_atr(df['High'], df['Low'], df['Close'])
    df['MA_5_Dist'] = df['Close'] / df['Close'].rolling(window=5).mean() - 1
    df['MA_20_Dist'] = df['Close'] / df['Close'].rolling(window=20).mean() - 1
    df['MACD'] = calculate_macd(df['Close'])
    df['BB_Pos'] = calculate_bollinger(df['Close'])
    df['RSI_14'] = calculate_rsi(df['Close'])
    df['Mom_10'] = df['Close'] / df['Close'].shift(10) - 1
    df['Volume_Norm'] = df['Volume'] / df['Volume'].rolling(window=20).mean()  # New: Normalized Volume
    
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

# Custom Loss (Increased penalty for sign mismatch)
def direction_aware_loss(y_true, y_pred):
    mse = K.mean(K.square(y_true - y_pred))
    penalty_weight = 1.0  # Increased from 0.5
    sign_mismatch = K.maximum(0.0, -1.0 * y_true * y_pred)
    return mse + penalty_weight * K.mean(sign_mismatch)

# MoE Model (Larger LSTM, lower reg)
class AdvancedGate(Layer):
    def __init__(self, n_experts=3, reg_val=1e-5, **kwargs):  # Lower reg
        super().__init__(**kwargs)
        self.n = n_experts
        self.lstm = LSTM(32, return_sequences=True, kernel_regularizer=l2(reg_val))
        self.attn = Attention()
        self.pool = GlobalAveragePooling1D()
        self.dropout = Dropout(0.2)
        self.dense = Dense(n_experts, activation='softmax', kernel_regularizer=l2(reg_val))

    def call(self, inputs):
        x = self.lstm(inputs)
        x = self.attn([x, x])
        x = self.pool(x)
        x = self.dropout(x)
        return self.dense(x)

def build_moe_model(input_shape):
    inputs = Input(shape=input_shape)
    reg_val = 1e-5  # Lowered
    
    # Expert 1: Trend (Larger LSTM)
    x1 = LSTM(128, return_sequences=True, kernel_regularizer=l2(reg_val))(inputs)  # 64 -> 128
    x1 = BatchNormalization()(x1)
    x1 = Dropout(0.3)(x1)
    x1 = LSTM(64, return_sequences=False, kernel_regularizer=l2(reg_val))(x1)  # 32 -> 64
    x1 = BatchNormalization()(x1)
    out1 = Dense(1, kernel_regularizer=l2(reg_val))(x1)
    
    # Expert 2: Volatility (CNN)
    x2 = Conv1D(32, 3, padding='same', kernel_regularizer=l2(reg_val))(inputs)
    x2 = Activation('relu')(x2)
    x2 = GlobalAveragePooling1D()(x2)
    x2 = Dropout(0.3)(x2)
    out2 = Dense(1, kernel_regularizer=l2(reg_val))(x2)
    
    # Expert 3: Mean Reversion (MLP)
    x3 = GlobalAveragePooling1D()(inputs)
    x3 = Dense(16, activation='relu', kernel_regularizer=l2(reg_val))(x3)
    x3 = Dropout(0.3)(x3)
    out3 = Dense(1, kernel_regularizer=l2(reg_val))(x3)
    
    gate_weights = AdvancedGate(n_experts=3, reg_val=reg_val)(inputs)
    
    stack = Lambda(lambda x: tf.stack(x, axis=1))([out1, out2, out3])
    gate_exp = Lambda(lambda x: tf.expand_dims(x, axis=-1))(gate_weights)
    mult = Multiply()([stack, gate_exp])
    out = Lambda(lambda x: tf.reduce_sum(x, axis=1))(mult)
    
    train_model = Model(inputs, out)
    opt = Adam(learning_rate=0.0005, clipnorm=1.0)
    train_model.compile(optimizer=opt, loss=direction_aware_loss, metrics=['mae'])
    
    predict_model = Model(inputs, [out, gate_weights])
    
    return train_model, predict_model

# Baselines (Unchanged)
def build_xgboost_baseline():
    return GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

class SimpleTransformer(layers.Layer):
    def __init__(self, reg_val=1e-5):  # Lower reg
        super().__init__()
        self.attn = Attention()
        self.dense1 = Dense(32, activation='relu', kernel_regularizer=l2(reg_val))
        self.dropout = Dropout(0.3)
        self.dense_out = Dense(1, kernel_regularizer=l2(reg_val))

    def call(self, inputs):
        x = self.attn([inputs, inputs])
        x = GlobalAveragePooling1D()(x)
        x = self.dense1(x)
        x = self.dropout(x)
        return self.dense_out(x)

def build_transformer_model(input_shape):
    inputs = Input(shape=input_shape)
    out = SimpleTransformer()(inputs)
    model = Model(inputs, out)
    opt = Adam(learning_rate=0.0005)
    model.compile(optimizer=opt, loss='mse', metrics=['mae'])
    return model

# Rolling Validation (seq_length=20, batch=16, epochs=50, tuned thresholds)
def run_rolling_analysis(df, train_window=756, test_window=63, step_size=63):
    target_col = 'Target_Ret'
    feature_cols = ['Log_Ret', 'Vol_20', 'ATR_14', 'MA_5_Dist', 'MA_20_Dist', 'MACD', 'RSI_14', 'Mom_10', 'BB_Pos', 'Volume_Norm']  # Added
    
    cols = [target_col] + feature_cols
    data_values = df[cols].values
    
    results = []
    predictions_list = []
    seq_length = 20  # Increased from 10
    transaction_cost = 0.0015
    
    total_samples = len(data_values)
    start_index = 0
    iteration = 1
    
    print("\nStarting Improved Rolling Validation...")
    print("-" * 100)
    print(f"{'Fold':<6} | {'RMSE':<8} | {'Acc':<8} | {'Net Return':<12} | {'Turnover':<10} | {'Sharpe':<8}")
    print("-" * 100)
    
    while start_index + train_window + test_window + seq_length < total_samples:
        train_start = start_index
        train_end = start_index + train_window
        test_start = train_end
        test_end = train_end + test_window
        
        train_raw = data_values[train_start:train_end]
        test_raw = data_values[test_start:test_end]
        
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_raw)
        test_scaled = scaler.transform(test_raw)
        
        full_test_context = np.concatenate((train_scaled[-seq_length:], test_scaled))
        X_train, y_train = create_sequences(train_scaled, seq_length, 0)
        X_test, y_test = create_sequences(full_test_context, seq_length, 0)
        
        if len(X_train) == 0 or len(X_test) == 0: break

        train_model, predict_model = build_moe_model((seq_length, len(cols)))
        es = EarlyStopping(monitor='loss', patience=8, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, min_lr=1e-5)
        
        train_model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0, callbacks=[es, reduce_lr])  # Tuned
        
        preds_scaled, gate_weights = predict_model.predict(X_test, verbose=0)
        preds_scaled = preds_scaled.flatten()
        
        dummy_pred = np.zeros((len(preds_scaled), len(cols)))
        dummy_pred[:, 0] = preds_scaled
        preds_inv = scaler.inverse_transform(dummy_pred)[:, 0]
        
        dummy_actual = np.zeros((len(y_test), len(cols)))
        dummy_actual[:, 0] = y_test.flatten()
        actual_inv = scaler.inverse_transform(dummy_actual)[:, 0]
        
        rmse = np.sqrt(mean_squared_error(actual_inv, preds_inv))
        acc = np.mean(np.sign(preds_inv) == np.sign(actual_inv))
        
        max_gate_weights = np.max(gate_weights, axis=1)
        positions = np.where(preds_inv > 0,
                             np.where(max_gate_weights > 0.7, 1.0,  # Lowered from 0.8 for more full positions
                                      np.where(max_gate_weights > 0.5, 0.5, 0.3)),
                             0.0)
        
        daily_strat_ret = positions * actual_inv
        
        position_changes = np.sum(np.diff(positions != 0) != 0)
        turnover = position_changes / len(positions) if len(positions) > 0 else 0
        
        costs = np.sum(np.abs(np.diff(positions)) * transaction_cost)
        net_period_return = np.sum(daily_strat_ret) - costs
        
        strat_ret_mean = np.mean(daily_strat_ret)
        strat_ret_std = np.std(daily_strat_ret)
        sharpe = strat_ret_mean / strat_ret_std if strat_ret_std != 0 else 0
        
        results.append({
            'Fold': iteration,
            'RMSE': rmse,
            'Direction_Acc': acc,
            'Net_Period_Return': net_period_return,
            'Turnover': turnover,
            'Sharpe': sharpe
        })
        
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        
        xgb_model = build_xgboost_baseline()
        xgb_model.fit(X_train_flat, y_train)
        xgb_preds = xgb_model.predict(X_test_flat)
        xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_preds))
        xgb_acc = np.mean(np.sign(xgb_preds) == np.sign(y_test))
        
        trans_model = build_transformer_model((seq_length, len(cols)))
        trans_model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0, callbacks=[es, reduce_lr])
        trans_preds = trans_model.predict(X_test, verbose=0).flatten()
        trans_rmse = np.sqrt(mean_squared_error(y_test, trans_preds))
        trans_acc = np.mean(np.sign(trans_preds) == np.sign(y_test))
        
        print(f"MoE Fold {iteration}: RMSE={rmse:.5f}, Acc={acc:.2%}, Net Return={net_period_return:.2%}, Turnover={turnover:.2%}, Sharpe={sharpe:.2f}")
        print(f"XGBoost: RMSE={xgb_rmse:.5f}, Acc={xgb_acc:.2%}")
        print(f"Transformer: RMSE={trans_rmse:.5f}, Acc={trans_acc:.2%}")
        
        fold_dates = df.index[test_start:test_start+len(y_test)]
        for date, pred, act in zip(fold_dates, preds_inv, actual_inv):
            predictions_list.append({'Date': date, 'Pred': pred, 'Actual': act})
            
        tf.keras.backend.clear_session()
        gc.collect()
        
        start_index += step_size
        iteration += 1
        
    return pd.DataFrame(results), pd.DataFrame(predictions_list)

# Visualization (Unchanged)
def plot_results(metrics_df, preds_df):
    plt.style.use('seaborn-v0_8-whitegrid')
    
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 3, 1)
    sns.histplot(metrics_df['Direction_Acc'], kde=True, color='green', bins=10)
    plt.axvline(0.5, color='red', linestyle='--')
    plt.title('Directional Accuracy (3-Day Trend)', fontsize=14)
    plt.xlabel('Accuracy')
    
    plt.subplot(1, 3, 2)
    colors = ['#d62728' if x >= 0 else 'green' for x in metrics_df['Net_Period_Return']]
    plt.bar(metrics_df['Fold'], metrics_df['Net_Period_Return'], color=colors)
    plt.axhline(0, color='black')
    plt.title('Net Strategy Return per Fold (Long-Only, with Costs)', fontsize=14)
    
    plt.subplot(1, 3, 3)
    sns.histplot(metrics_df['Turnover'], kde=True, color='blue', bins=10)
    plt.title('Turnover Rate Distribution', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    preds_df = preds_df.set_index('Date').sort_index()
    preds_df['Cum_Benchmark'] = np.cumsum(preds_df['Actual'])
    preds_df['Strategy_Ret'] = np.where(preds_df['Pred'] > 0, preds_df['Actual'], 0)
    preds_df['Cum_Strategy'] = np.cumsum(preds_df['Strategy_Ret'])
    
    plt.figure(figsize=(12, 6))
    plt.plot(preds_df.index, preds_df['Cum_Benchmark'], label='Market Trend (Benchmark)', color='gray', linestyle='--')
    plt.plot(preds_df.index, preds_df['Cum_Strategy'], label='MoE Strategy (Long-Only)', color='#d62728', linewidth=2)
    plt.title('Long-Only Strategy Performance', fontsize=16)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    data_file = '2.csi300_akshare_data.csv'
    if os.path.exists(data_file):
        df = load_and_process_data(data_file)
        if df is not None:
            metrics, preds = run_rolling_analysis(df)
            print("\nSummary:")
            print(metrics.describe())
            plot_results(metrics, preds)
    else:
        print("Data file not found.")