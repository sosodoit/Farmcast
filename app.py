import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------ 전처리 & 시각화 함수 ------------------
def preprocess_data(df):
    cutoff_date = pd.to_datetime('2020-09-28')
    cols_to_zero = ['cabbage', 'radish', 'garlic', 'onion', 'daikon', 'cilantro', 'artichoke']
    df.loc[df.index > cutoff_date, cols_to_zero] = np.nan
    return df

def plot_predictions_over_time(df, vegetables, rolling_mean_window):
    fig, ax = plt.subplots(figsize=(14, 7))

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    num_colors = len(colors)

    for i, veg in enumerate(vegetables):
        ax.plot(df.index, df[veg], label=veg, linewidth=2, color=colors[i % num_colors])
        rolling_mean = df[veg].rolling(window=rolling_mean_window).mean()
        ax.plot(df.index, rolling_mean, label=f'{veg} ({rolling_mean_window}-day Rolling Mean)', linestyle='--', color=colors[i % num_colors])

    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel('Price', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, color='lightgrey', linestyle='--')
    fig.tight_layout()
    st.pyplot(fig)

# ------------------ 데이터 로딩 ------------------
csv_file_path = 'data/streamlit_data.csv'
metric_file_path = 'data/metric_summary.csv'

@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    return df

@st.cache_data
def load_metrics(file_path):
    metrics = pd.read_csv(file_path)
    metrics.set_index('product', inplace=True)
    return metrics

# 데이터 로드
df = load_data(csv_file_path)
metric_summary = load_metrics(metric_file_path)

# ------------------ 제목 ------------------
st.title('🥬 농산물 가격 분석 대시보드')
st.markdown("가격 데이터에 대한 통계적 특성 및 계절성을 분석합니다.")
st.markdown("""왼쪽에서 품목과 예측모델, 날짜를 입력하면 특정기간 이후 예측 가격이 표시됩니다.""")

# ------------------ 사이드바 ------------------
st.sidebar.header('📅 조회 기간')
start_date = st.sidebar.date_input("시작일", df.index.min())
end_date = st.sidebar.date_input("종료일", df.index.max())

st.sidebar.header('🥦 품목 선택')
sorted_vegetables = sorted(df.columns)
vegetables = st.sidebar.multiselect('조회 품목:', sorted_vegetables)

st.sidebar.header("📈 Rolling Mean Window")
rolling_mean_window = st.sidebar.slider('Rolling Mean Window', min_value=1, max_value=30, value=7)

st.sidebar.markdown("""
| Korean | English    |
|--------|------------|
| 배추   | cabbage    |
| 무     | radish     |
| 마늘   | garlic     |
| 양파   | onion      |
| 대파   | daikon     |
| 건고추 | cilantro   |
| 깻잎   | artichoke  |
""")

# ------------------ 데이터 필터링 ------------------
df = preprocess_data(df)
filtered_df = df.loc[start_date:end_date]

# ------------------ 탭 구조 ------------------
tab1, tab2 = st.tabs(["📈 데이터 시각화", "🤖 예측 결과 시각화"])

# ------------------ Tab 1: 데이터 시각화 ------------------
with tab1:
    st.header("📊 농산물 가격 데이터 시각화")

    if vegetables:
        st.subheader("1️⃣ 시계열 가격 추세 (Raw + Rolling Mean)")
        fig, ax = plt.subplots(figsize=(14, 6))
        for i, veg in enumerate(vegetables):
            ax.plot(filtered_df.index, filtered_df[veg], label=veg)
            ax.plot(filtered_df.index, filtered_df[veg].rolling(rolling_mean_window).mean(), 
                    linestyle='--', label=f"{veg} {rolling_mean_window}일 평균")
        ax.set_ylabel("가격")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        st.subheader("2️⃣ 가격 분포 (Histogram + KDE)")
        fig, axs = plt.subplots(1, len(vegetables), figsize=(5*len(vegetables), 4))
        if len(vegetables) == 1:
            axs = [axs]
        for i, veg in enumerate(vegetables):
            sns.histplot(filtered_df[veg].dropna(), kde=True, ax=axs[i], color='skyblue')
            axs[i].set_title(f"{veg}")
        st.pyplot(fig)

        st.subheader("3️⃣ 상관관계 히트맵")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(filtered_df[vegetables].corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

        st.subheader("4️⃣ 요일별 평균 가격")
        temp_df = filtered_df.copy()
        temp_df['day_of_week'] = temp_df.index.dayofweek
        weekday_mean = temp_df.groupby('day_of_week')[vegetables].mean()
        weekday_mean.index = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
        st.bar_chart(weekday_mean)

        st.subheader("5️⃣ 월별 평균 가격")
        temp_df['month'] = temp_df.index.month
        month_mean = temp_df.groupby('month')[vegetables].mean()
        st.line_chart(month_mean)
    
    if st.checkbox("📄 원본 데이터 보기 (Tab1)"):
        st.write(filtered_df)

# ------------------ Tab 2: 예측 결과 시각화 ------------------
with tab2:
    st.header("📊 농산물 가격 예측 결과")

    st.subheader("1️⃣ 예측 결과 시계열")
    if vegetables:
        fig, ax = plt.subplots(figsize=(14, 6))
        for veg in vegetables:
            if veg in df.columns:
                ax.plot(df.index, df[veg], label=f"{veg} 실제")
                ax.plot(df.index, df[veg].rolling(rolling_mean_window).mean(), linestyle='--', label=f"{veg} 예측값(가정)")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
    else:
        st.warning("왼쪽 사이드바에서 품목을 선택해주세요.")

    st.subheader("2️⃣ 모델 정확도 요약 (MdAPE 기반)")
    st.dataframe(metric_summary)

    if st.checkbox("📄 원본 데이터 보기 (Tab2)"):
        st.write(df)