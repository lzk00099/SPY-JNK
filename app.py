import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# 页面配置
st.set_page_config(page_title="JNK vs SPY 乖离率追踪器", layout="wide")
st.title("JNK/SPY 乖离率与并轨追踪模型")

# 侧边栏设置
st.sidebar.header("参数设置")
period = st.sidebar.selectbox("时间范围", ["5d", "1mo", "3mo", "6mo", "1y"], index=0)
interval = st.sidebar.selectbox("K线级别", ["5m", "15m", "30m", "1h", "1d"], index=1)
rolling_window = st.sidebar.slider("滚动回归窗口(期数)", 10, 100, 20)

@st.cache_data(ttl=300) # 缓存5分钟
def load_data(period, interval):
    tickers = ["SPY", "JNK"]
    data = yf.download(tickers, period=period, interval=interval)['Close']
    data = data.dropna()
    return data

try:
    df = load_data(period, interval)
    
    if df.empty:
        st.error("获取数据失败，请尝试更改时间范围或K线级别（例如 5m 数据只能获取最近 60 天）。")
    else:
        # 计算归一化走势 (以起点为100)
        df_normalized = (df / df.iloc[0]) * 100
        
        # 计算动态比值和 Z-Score
        df['Ratio'] = df['SPY'] / df['JNK']
        df['Ratio_Mean'] = df['Ratio'].rolling(window=rolling_window).mean()
        df['Ratio_STD'] = df['Ratio'].rolling(window=rolling_window).std()
        df['Z_Score'] = (df['Ratio'] - df['Ratio_Mean']) / df['Ratio_STD']
        
        # 计算 SPY 理论应有价格 (基于过去窗口的 JNK/SPY 均值比率)
        df['Implied_SPY'] = df['JNK'] * df['Ratio_Mean']
        df['Spread'] = df['Implied_SPY'] - df['SPY']

        # 获取最新数据
        latest_spy = df['SPY'].iloc[-1]
        latest_jnk = df['JNK'].iloc[-1]
        implied_spy = df['Implied_SPY'].iloc[-1]
        spread = df['Spread'].iloc[-1]

        # 顶部指标卡
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("SPY 现价", f"{latest_spy:.2f}")
        col2.metric("JNK 现价", f"{latest_jnk:.2f}")
        col3.metric("SPY 理论价格 (基于JNK)", f"{implied_spy:.2f}", f"{spread:.2f} 差值")
        
        signal = "SPY 估值过低 (看多/并轨预期)" if spread > 0 else "SPY 估值过高 (看空)"
        color = "normal" if spread > 0 else "inverse"
        col4.metric("当前套利信号", signal, delta_color=color)

        # 图表 1：SPY 与理论价格对比
        st.subheader("SPY 实际价格 vs 理论价格 (基于 JNK 估算)")
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=df.index, y=df['SPY'], mode='lines', name='SPY 实际价格', line=dict(color='blue')))
        fig1.add_trace(go.Scatter(x=df.index, y=df['Implied_SPY'], mode='lines', name='SPY 理论价格', line=dict(color='orange', dash='dash')))
        fig1.update_layout(height=400, template="plotly_white", hovermode="x unified")
        st.plotly_chart(fig1, use_container_width=True)

        # 图表 2：Z-Score 乖离率
        st.subheader("JNK/SPY 乖离率 Z-Score")
        st.write("当 Z-Score 低于 -1.5 时，表明 SPY 相对于 JNK 被严重低估，是做多末日/周度 Call 的信号区域。")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df.index, y=df['Z_Score'], mode='lines', name='Z-Score', fill='tozeroy'))
        fig2.add_hline(y=1.5, line_dash="dash", line_color="red", annotation_text="SPY超买区")
        fig2.add_hline(y=-1.5, line_dash="dash", line_color="green", annotation_text="SPY超卖区 (介入做多)")
        fig2.update_layout(height=300, template="plotly_white")
        st.plotly_chart(fig2, use_container_width=True)

except Exception as e:
    st.error(f"发生错误: {e}")
