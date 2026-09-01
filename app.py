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
interval = st.sidebar.selectbox("K线级别", ["5m", "15m", "30m", "1h", "1d"], index=4) # 默认选 1d
rolling_window = st.sidebar.slider("滚动回归窗口(期数)", 10, 100, 20)

# 手动刷新按钮
if st.sidebar.button("🔄 立即刷新盘中实时数据"):
    st.cache_data.clear()

# 盘中将缓存设为 60 秒 (1分钟自动过期)
@st.cache_data(ttl=60)
def load_data(period, interval):
    tickers = ["SPY", "JNK"]
    # 1. 抓取历史日线数据
    data = yf.download(tickers, period=period, interval=interval)['Close']
    data = data.dropna()
    
    # 2. 强行抓取盘中最新的 1分钟线 收盘价 (比 fast_info 稳定100倍)
    try:
        live_data = yf.download(tickers, period="1d", interval="1m")['Close']
        if not live_data.empty:
            spy_live = live_data['SPY'].dropna().iloc[-1]
            jnk_live = live_data['JNK'].dropna().iloc[-1]
            
            if interval == "1d":
                # 获取美东时间的今天，并将其格式化为 00:00:00，去除时区以完美匹配 yfinance 的日线索引
                today_est = pd.Timestamp.now(tz="US/Eastern").normalize().tz_localize(None)
                last_idx = data.index[-1]
                
                # 检查 yfinance 历史数据是否带有时区，如果有，就给 today 也补上相同格式
                if last_idx.tz is not None:
                     today_est = pd.Timestamp.now(tz="US/Eastern").normalize().tz_localize(last_idx.tz)

                # 核心覆盖逻辑：如果历史数据的最后一天已经是今天，直接用刚才抓的盘中现价覆盖；如果不是，就新增今天这一行
                if last_idx.normalize() == today_est:
                    data.loc[last_idx, 'SPY'] = spy_live
                    data.loc[last_idx, 'JNK'] = jnk_live
                else:
                    data.loc[today_est, 'SPY'] = spy_live
                    data.loc[today_est, 'JNK'] = jnk_live
            else:
                # 如果用户选的本身就是分钟线(15m, 30m等)，直接把 1分钟线 的最新一秒时间戳塞进去
                now_idx = live_data.index[-1]
                data.loc[now_idx, 'SPY'] = spy_live
                data.loc[now_idx, 'JNK'] = jnk_live
                
    except Exception as e:
        # 如果因为网络问题抓取实时数据失败，在侧边栏显示警告，方便排查
        st.sidebar.warning(f"⚠️ 实时现价抓取失败，退回历史收盘价。报错: {e}")
        
    return data
try:
    df = load_data(period, interval)
    
    if df.empty:
        st.error("获取数据失败，请尝试更改时间范围或K线级别。")
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

        # 获取最新数据 (包含刚才插入的实时现价)
        latest_spy = df['SPY'].iloc[-1]
        latest_jnk = df['JNK'].iloc[-1]
        implied_spy = df['Implied_SPY'].iloc[-1]
        spread = df['Spread'].iloc[-1]
        latest_z = df['Z_Score'].iloc[-1]

        # 顶部指标卡
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("SPY 实时价格", f"{latest_spy:.2f}")
        col2.metric("JNK 实时价格", f"{latest_jnk:.2f}")
        col3.metric("SPY 理论估值", f"{implied_spy:.2f}", f"{spread:+.2f} 差值")
        
        signal = f"超卖做多 (Z: {latest_z:.2f})" if latest_z < -1.5 else (f"超买做空 (Z: {latest_z:.2f})" if latest_z > 1.5 else f"区间中性 (Z: {latest_z:.2f})")
        color = "normal" if latest_z < -1.5 else ("inverse" if latest_z > 1.5 else "off")
        col4.metric("实时信号状态", signal, delta_color=color)

        # 创建 2 行 1 列的子图结构，共享 X 轴
        fig = make_subplots(
            rows=2, 
            cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.08,
            subplot_titles=("SPY 实际价格 vs 理论价格 (基于 JNK)", "JNK/SPY 乖离率 Z-Score"),
            row_heights=[0.6, 0.4]
        )

        # 1. 绘制上图：SPY 实际价格 vs 理论价格
        fig.add_trace(
            go.Scatter(x=df.index, y=df['SPY'], mode='lines', name='SPY 实际价格', line=dict(color='#1f77b4')), 
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['Implied_SPY'], mode='lines', name='SPY 理论价格', line=dict(color='#ff7f0e', dash='dash')), 
            row=1, col=1
        )

        # 2. 绘制下图：Z-Score 乖离率
        fig.add_trace(
            go.Scatter(x=df.index, y=df['Z_Score'], mode='lines', name='Z-Score', fill='tozeroy', line=dict(color='#2ca02c')), 
            row=2, col=1
        )

        # 添加下图超买超卖阈值线
        fig.add_hline(y=1.5, line_dash="dash", line_color="red", annotation_text="超买", row=2, col=1)
        fig.add_hline(y=-1.5, line_dash="dash", line_color="green", annotation_text="超卖", row=2, col=1)

        # 统一布局调整：图例置底 + 横向排列
        fig.update_layout(
            height=650,
            template="plotly_white",
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.18,
                xanchor="center",
                x=0.5
            ),
            margin=dict(l=20, r=20, t=40, b=60)
        )

        # 渲染图表
        st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"发生错误: {e}")
