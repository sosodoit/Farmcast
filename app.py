import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ------------------ 레이아웃 구성 ------------------
st.set_page_config(layout="wide")

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

# 한글 매핑 딕셔너리
veggie_kor = {
    'cabbage': '배추', 'radish': '무', 'garlic': '마늘',
    'onion': '양파', 'daikon': '대파', 'cilantro': '건고추', 'artichoke': '깻잎'
}

day_map = {
    0: '월', 1: '화', 2: '수', 3: '목', 4: '금', 5: '토', 6: '일'
}

# ------------------ 제목 ------------------
with st.container():
    st.markdown("""
    <div style="background-color:#f0f4f8; padding: 20px 25px; border-radius: 12px; border: 1px solid #dfe6ec">
        <h3 style='color: #174c88;'>농산물 가격 인사이트 대시보드</h2>
        <p style='font-size: 16px; color: #333;'>
        주요 농산물의 가격 데이터를 시각화하고, 시간 흐름에 따른 추세와 예측 결과를 확인할 수 있습니다.
        </p>
    </div>
    <br>
    """, unsafe_allow_html=True)

# ------------------ 사이드바 ------------------

st.sidebar.header('조회 기간')
start_date = st.sidebar.date_input("시작일", df.index.min())
end_date = st.sidebar.date_input("종료일", df.index.max())

st.sidebar.header('옵션 선택')
main_vegetables = ['cabbage', 'radish', 'garlic', 'onion', 'daikon', 'cilantro', 'artichoke']
# sorted_vegetables = sorted(df.columns)

# 분석 품목 선택 
sorted_vegetables = sorted([col for col in df.columns if '_pred_' not in col and col != 'date'])
selected_vegetables = st.sidebar.multiselect(
    "분석할 품목 선택",
    options=sorted_vegetables,
    default=main_vegetables
)

# selected_vegetables = st.sidebar.multiselect(
#     '분석할 품목 선택',
#     options=main_vegetables,
#     format_func=lambda x: veggie_kor[x],
#     default=main_vegetables
# )

# 시계열 유형 선택
available_series = ['Actual', 'Predicted']
selected_series = st.sidebar.multiselect(
    '시계열 유형 선택',
    options=available_series,
    default=['Actual']
)

# 예측 모델 선택
available_models = ['LGBM', 'MLP', 'RandomForest', 'Ridge', 'XGBoost', 'average', 'stack']
model_kor = {
    'LGBM': 'LGBM', 'MLP': 'MLP', 'RandomForest': '랜덤포레스트',
    'Ridge': '릿지', 'XGBoost': '엑스지부스트', 'average': '보팅', 'stack': '스태킹'
}

if 'Predicted' in selected_series:
    selected_models = st.sidebar.multiselect(
        "🤖 예측 모델 선택",
        options=available_models,
        default=['average']
    )

else:
    selected_models = [] 

st.sidebar.header("Rolling Mean Window")
rolling_mean_window = st.sidebar.slider('이동 평균 계산', min_value=1, max_value=30, value=7)

# ------------------ 데이터 필터링 ------------------
df = preprocess_data(df)
filtered_df = df.loc[start_date:end_date]

# ------------------ 탭 구조 ------------------
tab1, tab2 = st.tabs(["📈 데이터 시각화", "🤖 예측 결과 시각화"])

# ------------------ Tab 1: 데이터 시각화 ------------------
with tab1:

    def highlight_selected(row):
        if row['품목'] == selected_kor:
            return ['background-color: #e0f3ff'] * len(row)
        else:
            return [''] * len(row)
        
    col1, col2 = st.columns(2)
    # 세션 상태로 현재 인덱스 저장
    if 'veg_index' not in st.session_state:
        st.session_state.veg_index = 0

    with col1:
        # ------------------ 기술통계 (1행 1열) ------------------
        st.markdown("""<h3 style='font-size: 20px;'>1️⃣ 기술 통계</h3>""", unsafe_allow_html=True)
        summary = filtered_df[main_vegetables].describe().T[['mean', 'std', 'min', 'max']].reset_index()
        summary['index'] = summary['index'].map(veggie_kor)
        summary.columns = ['품목', '평균', '표준편차', '최소값', '최대값']
        # st.dataframe(summary, use_container_width=True, hide_index=True)
        selected_kor = veggie_kor[main_vegetables[st.session_state.veg_index]]
        styled_summary = summary.style.apply(highlight_selected, axis=1)
        st.dataframe(styled_summary, use_container_width=True, hide_index=True)

    with col2:     
        # ------------------ 가격분포 (1행 2열) ------------------ 
        left_col, right_col = st.columns([8,1])

        with left_col:
            selected_veg = main_vegetables[st.session_state.veg_index]
            st.markdown(f"<h3 style='font-size: 20px;'>2️⃣ 가격 분포: {veggie_kor[selected_veg]}</h3>", unsafe_allow_html=True)

        with right_col:            
            if st.button(" ▶️ ", help="다음 품목 보기"):                
                st.session_state.veg_index = (st.session_state.veg_index + 1) % len(main_vegetables)
                
        # 가격 분포 그래프
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(filtered_df[selected_veg].dropna(), kde=True, ax=ax, color='skyblue')
        ax.set_xlabel("가격")
        ax.set_ylabel("밀도")
        ax.grid(True)
        st.pyplot(fig)        

    st.markdown("""
    <hr style='
        border: none;
        height: 2px;
        background-color:rgba(49, 51, 63, 0.1);
        margin: 20px 0;
        border-radius: 4px;'
    >
    """, unsafe_allow_html=True)

    # ------------------ 시각화 더보기 ------------------
    st.markdown("시각화 더보기")
    if st.checkbox("시계열 가격 추세"):
        st.markdown("""<h3 style='font-size: 20px;'>시계열 가격 추세</h3>""", unsafe_allow_html=True)

        if selected_vegetables and selected_series:
            plot_predictions_over_time(filtered_df, selected_vegetables, rolling_mean_window)
            
        else:
            st.info("왼쪽에서 품목과 시계열 항목을 선택해주세요.")

    if st.checkbox("상관관계 히트맵"):
        if selected_vegetables:
            st.markdown("""<h3 style='font-size: 20px;'>상관관계 히트맵</h3>""", unsafe_allow_html=True)

            if len(selected_vegetables) >= 2:
                corr_matrix = filtered_df[selected_vegetables].corr()
                corr_matrix.index = [veggie_kor[v] for v in corr_matrix.index]
                corr_matrix.columns = [veggie_kor[v] for v in corr_matrix.columns]

                fig, ax = plt.subplots(figsize=(6, 4))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax, vmin=-1, vmax=1)
                st.pyplot(fig)

            else:
                st.warning("두 개 이상의 품목을 선택해주세요.")

    if st.checkbox("요일별 평균 가격"):
        if selected_vegetables:
            st.markdown("""<h3 style='font-size: 20px;'>요일별 평균 가격</h3>""", unsafe_allow_html=True)
            
            # 요일 컬럼 추가
            temp_df = filtered_df.copy()
            temp_df['day'] = temp_df.index.dayofweek # 0~6 (월~일)
            
            # melt 형태로 변환
            melted = temp_df[selected_vegetables + ['day']].melt(id_vars='day', var_name='품목', value_name='가격')
            melted.dropna(inplace=True)
            melted['요일'] = melted['day'].map(day_map)
            melted['품목'] = melted['품목'].map(veggie_kor)

            # 시각화
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(data=melted, x='품목', y='가격', hue='요일', ax=ax)
            ax.grid(True, axis='y', linestyle='--', alpha=0.5)
            ax.legend(title='요일')
            st.pyplot(fig)

    if st.checkbox("월별 평균 가격"):
        if selected_vegetables:
            st.markdown("""<h3 style='font-size: 20px;'>월별 평균 가격</h3>""", unsafe_allow_html=True)

            # 월 컬럼 추가
            temp_df = filtered_df.copy()
            temp_df['month'] = temp_df.index.month

            # 월별 평균 가격 계산
            month_mean = temp_df.groupby('month')[selected_vegetables].mean().T
            month_mean.index = month_mean.index.map(veggie_kor)  # 한글 이름 매핑

            # melt → 그래프 그리기 편하게 변환
            melted_month = month_mean.reset_index().melt(id_vars='index', var_name='월', value_name='평균 가격')
            melted_month.rename(columns={'index': '품목'}, inplace=True)

            # 시각화
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(data=melted_month, x="품목", y="평균 가격", hue="월", palette="Set3", ax=ax)
            ax.grid(True, linestyle='--', alpha=0.5)
            st.pyplot(fig)

# ------------------ Tab 2: 예측 결과 시각화 ------------------
with tab2:

    st.markdown("""<h3 style='font-size: 20px;'>1️⃣ 예측 결과 시계열</h3>""", unsafe_allow_html=True)
    
    filtered_columns = []

    if 'Predicted' in selected_series:
        for veg in selected_vegetables:
            filtered_columns.append(veg)
            for model in selected_models:
                col = f"{veg}_pred_{model}"
                if col in filtered_df.columns:
                    filtered_columns.append(col)

    if filtered_columns:
        plot_predictions_over_time(filtered_df, filtered_columns, rolling_mean_window)

    else:
        st.warning("왼쪽 사이드바에서 품목을 선택해주세요.")

    st.markdown("""<h3 style='font-size: 20px;'>2️⃣ 모델 정확도 요약 (MdAPE 기반)</h3>""", unsafe_allow_html=True)
    
    st.dataframe(metric_summary)

    if st.checkbox("원본 데이터 보기"):
        st.write(filtered_df)