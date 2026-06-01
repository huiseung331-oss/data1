import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ── 페이지 설정 ──────────────────────────────────────────────
st.set_page_config(
    page_title="서울 기온 상승 분석 (1907–2026)",
    page_icon="🌡️",
    layout="wide",
)

# ── 한글 폰트 설정 ───────────────────────────────────────────
@st.cache_resource
def set_korean_font():
    import subprocess, os
    subprocess.run(["apt-get", "install", "-y", "fonts-nanum"], capture_output=True)
    font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        plt.rcParams["font.family"] = "NanumGothic"
    plt.rcParams["axes.unicode_minus"] = False

set_korean_font()

# ── 데이터 로드 ──────────────────────────────────────────────
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = ["날짜", "지점", "평균기온", "최저기온", "최고기온"]
    df["날짜"] = pd.to_datetime(df["날짜"].astype(str).str.strip(), errors="coerce")
    df = df.dropna(subset=["날짜", "평균기온"])
    df["연도"] = df["날짜"].dt.year
    df["월"] = df["날짜"].dt.month
    df["계절"] = df["월"].map(
        lambda m: "봄(3-5월)" if m in [3, 4, 5]
        else "여름(6-8월)" if m in [6, 7, 8]
        else "가을(9-11월)" if m in [9, 10, 11]
        else "겨울(12-2월)"
    )
    return df

uploaded = st.sidebar.file_uploader("📂 CSV 파일 업로드 (기상청 형식)", type="csv")
if uploaded:
    df = load_data(uploaded)
else:
    try:
        df = load_data("ta_20260601093156.csv")
    except:
        st.error("CSV 파일을 업로드해 주세요 (사이드바).")
        st.stop()

# ── 사이드바 컨트롤 ──────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ 분석 설정")
cutoff = st.sidebar.slider("기준 연도 (이전/이후 분할)", 1950, 2000, 1980)
min_yr, max_yr = int(df["연도"].min()), int(df["연도"].max())
year_range = st.sidebar.slider("분석 연도 범위", min_yr, max_yr, (min_yr, max_yr))
season_filter = st.sidebar.multiselect(
    "계절 필터",
    ["봄(3-5월)", "여름(6-8월)", "가을(9-11월)", "겨울(12-2월)"],
    default=["봄(3-5월)", "여름(6-8월)", "가을(9-11월)", "겨울(12-2월)"],
)

# ── 데이터 필터링 ────────────────────────────────────────────
df_f = df[
    (df["연도"] >= year_range[0])
    & (df["연도"] <= year_range[1])
    & (df["계절"].isin(season_filter))
].copy()

# 연간 평균
annual = df_f.groupby("연도")["평균기온"].mean().reset_index()
annual.columns = ["연도", "연평균기온"]
annual["시기"] = annual["연도"].apply(lambda y: f"{cutoff}년 이전" if y < cutoff else f"{cutoff}년 이후")

before = annual[annual["연도"] < cutoff]["연평균기온"]
after  = annual[annual["연도"] >= cutoff]["연평균기온"]

# ── 헤더 ─────────────────────────────────────────────────────
st.title("🌡️ 서울 기온 상승 분석")
st.markdown(f"**가설:** {cutoff}년을 기점으로 서울의 기온 상승 속도가 빨라졌다")
st.markdown(f"분석 기간: {year_range[0]} – {year_range[1]}  |  데이터 출처: 기상청 지점 108 (서울)")
st.markdown("---")

# ── KPI 카드 ─────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)

slope_before, intercept_before, *_ = stats.linregress(
    annual[annual["연도"] < cutoff]["연도"],
    annual[annual["연도"] < cutoff]["연평균기온"]
) if len(before) > 1 else (0, 0, None, None, None)

slope_after, intercept_after, *_ = stats.linregress(
    annual[annual["연도"] >= cutoff]["연도"],
    annual[annual["연도"] >= cutoff]["연평균기온"]
) if len(after) > 1 else (0, 0, None, None, None)

t_stat, p_val = stats.ttest_ind(before, after) if len(before) > 1 and len(after) > 1 else (0, 1)

k1.metric(f"{cutoff}년 이전 평균", f"{before.mean():.2f}°C")
k2.metric(f"{cutoff}년 이후 평균", f"{after.mean():.2f}°C", f"+{after.mean()-before.mean():.2f}°C")
k3.metric(f"상승 속도 (이전)", f"{slope_before*10:.3f}°C/10년")
k4.metric(f"상승 속도 (이후)", f"{slope_after*10:.3f}°C/10년",
          f"{(slope_after-slope_before)*10:+.3f}°C/10년")

st.markdown("---")

# ── 탭 구성 ──────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["📈 연도별 추세", "📊 분포 비교", "🗓️ 계절별 분석", "🔬 통계 검정"]
)

COLOR_BEFORE = "#4A90D9"
COLOR_AFTER  = "#E74C3C"

# ── TAB 1: 연도별 추세 ────────────────────────────────────────
with tab1:
    fig, ax = plt.subplots(figsize=(13, 5))

    bef_df = annual[annual["연도"] < cutoff]
    aft_df = annual[annual["연도"] >= cutoff]

    ax.scatter(bef_df["연도"], bef_df["연평균기온"], color=COLOR_BEFORE, alpha=0.55, s=18, label=f"{cutoff}년 이전")
    ax.scatter(aft_df["연도"], aft_df["연평균기온"], color=COLOR_AFTER,  alpha=0.55, s=18, label=f"{cutoff}년 이후")

    # 추세선
    for sub, slope, intercept, color in [
        (bef_df, slope_before, intercept_before, COLOR_BEFORE),
        (aft_df, slope_after,  intercept_after,  COLOR_AFTER),
    ]:
        if len(sub) > 1:
            xs = sub["연도"].values
            ax.plot(xs, slope * xs + intercept, color=color, linewidth=2.2)

    # 이동평균
    ann_sorted = annual.sort_values("연도")
    ax.plot(ann_sorted["연도"], ann_sorted["연평균기온"].rolling(10, center=True).mean(),
            color="black", linewidth=1.5, linestyle="--", alpha=0.6, label="10년 이동평균")

    ax.axvline(cutoff, color="gray", linestyle=":", linewidth=1.5)
    ax.set_xlabel("연도"); ax.set_ylabel("연평균기온 (°C)")
    ax.set_title("서울 연평균기온 추세 및 선형 회귀")
    ax.legend(); ax.grid(alpha=0.3)
    st.pyplot(fig, use_container_width=True)

    st.info(
        f"**이전({year_range[0]}–{cutoff-1})** 기온 상승 속도: **{slope_before*10:.3f}°C/10년**  \n"
        f"**이후({cutoff}–{year_range[1]})** 기온 상승 속도: **{slope_after*10:.3f}°C/10년**  \n"
        f"→ 이후 기간에 약 **{abs(slope_after/slope_before):.1f}배** 빠른 상승"
        if slope_before != 0 else ""
    )

# ── TAB 2: 분포 비교 ──────────────────────────────────────────
with tab2:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # 히스토그램
    axes[0].hist(before, bins=20, color=COLOR_BEFORE, alpha=0.7, label=f"{cutoff}년 이전", edgecolor="white")
    axes[0].hist(after,  bins=20, color=COLOR_AFTER,  alpha=0.7, label=f"{cutoff}년 이후", edgecolor="white")
    axes[0].axvline(before.mean(), color=COLOR_BEFORE, linestyle="--", linewidth=1.8)
    axes[0].axvline(after.mean(),  color=COLOR_AFTER,  linestyle="--", linewidth=1.8)
    axes[0].set_xlabel("연평균기온 (°C)"); axes[0].set_ylabel("빈도")
    axes[0].set_title("연평균기온 분포")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    # 박스플롯
    bp = axes[1].boxplot(
        [before.dropna(), after.dropna()],
        labels=[f"{cutoff}년 이전", f"{cutoff}년 이후"],
        patch_artist=True,
        medianprops=dict(color="white", linewidth=2),
    )
    bp["boxes"][0].set_facecolor(COLOR_BEFORE)
    bp["boxes"][1].set_facecolor(COLOR_AFTER)
    axes[1].set_ylabel("연평균기온 (°C)"); axes[1].set_title("연평균기온 박스플롯")
    axes[1].grid(alpha=0.3)

    st.pyplot(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    col1.dataframe(
        pd.DataFrame({
            "통계량": ["평균", "중앙값", "표준편차", "최솟값", "최댓값", "데이터 수(년)"],
            f"{cutoff}년 이전": [f"{before.mean():.2f}°C", f"{before.median():.2f}°C",
                                   f"{before.std():.2f}°C", f"{before.min():.2f}°C",
                                   f"{before.max():.2f}°C", str(len(before))],
            f"{cutoff}년 이후": [f"{after.mean():.2f}°C", f"{after.median():.2f}°C",
                                   f"{after.std():.2f}°C", f"{after.min():.2f}°C",
                                   f"{after.max():.2f}°C", str(len(after))],
        }), hide_index=True, use_container_width=True
    )

# ── TAB 3: 계절별 분석 ────────────────────────────────────────
with tab3:
    seasons = ["봄(3-5월)", "여름(6-8월)", "가을(9-11월)", "겨울(12-2월)"]
    season_stats = []
    for s in seasons:
        sdf = df_f[df_f["계절"] == s].groupby("연도")["평균기온"].mean().reset_index()
        sdf.columns = ["연도", "평균기온"]
        b = sdf[sdf["연도"] < cutoff]["평균기온"]
        a = sdf[sdf["연도"] >= cutoff]["평균기온"]
        sl_b = stats.linregress(sdf[sdf["연도"] < cutoff]["연도"], b)[0] if len(b) > 1 else 0
        sl_a = stats.linregress(sdf[sdf["연도"] >= cutoff]["연도"], a)[0] if len(a) > 1 else 0
        season_stats.append({
            "계절": s,
            f"이전 평균(°C)": round(b.mean(), 2),
            f"이후 평균(°C)": round(a.mean(), 2),
            "기온 변화(°C)": round(a.mean() - b.mean(), 2),
            "이전 상승 속도(/10년)": round(sl_b * 10, 3),
            "이후 상승 속도(/10년)": round(sl_a * 10, 3),
        })
    season_df = pd.DataFrame(season_stats)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()
    colors = ["#2ECC71", "#E74C3C", "#F39C12", "#3498DB"]
    for i, s in enumerate(seasons):
        sdf = df_f[df_f["계절"] == s].groupby("연도")["평균기온"].mean().reset_index()
        sdf.columns = ["연도", "평균기온"]
        bef = sdf[sdf["연도"] < cutoff]
        aft = sdf[sdf["연도"] >= cutoff]
        ax = axes[i]
        ax.scatter(bef["연도"], bef["평균기온"], color=COLOR_BEFORE, alpha=0.5, s=14)
        ax.scatter(aft["연도"], aft["평균기온"], color=COLOR_AFTER,  alpha=0.5, s=14)
        for sub, color in [(bef, COLOR_BEFORE), (aft, COLOR_AFTER)]:
            if len(sub) > 1:
                sl, ic, *_ = stats.linregress(sub["연도"], sub["평균기온"])
                xs = sub["연도"].values
                ax.plot(xs, sl * xs + ic, color=color, linewidth=2)
        ax.axvline(cutoff, color="gray", linestyle=":", linewidth=1.2)
        ax.set_title(s); ax.set_xlabel("연도"); ax.set_ylabel("평균기온 (°C)")
        ax.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

    st.dataframe(season_df, hide_index=True, use_container_width=True)

# ── TAB 4: 통계 검정 ─────────────────────────────────────────
with tab4:
    st.subheader("독립표본 t-검정 (이전 vs 이후 연평균기온)")

    col1, col2 = st.columns([2, 1])
    with col1:
        result_df = pd.DataFrame({
            "항목": ["t-통계량", "p-값", "유의수준 0.05 기준", "결론"],
            "값": [
                f"{t_stat:.4f}",
                f"{p_val:.6f}",
                "✅ 유의함" if p_val < 0.05 else "❌ 유의하지 않음",
                f"{'두 기간의 평균기온 차이는 통계적으로 유의합니다.' if p_val < 0.05 else '통계적으로 유의한 차이가 없습니다.'}"
            ]
        })
        st.dataframe(result_df, hide_index=True, use_container_width=True)

    with col2:
        st.metric("평균기온 차이", f"{after.mean()-before.mean():.2f}°C")
        st.metric("p-값", f"{p_val:.6f}")
        st.metric("가설 채택 여부", "✅ 채택" if p_val < 0.05 else "❌ 기각")

    # 10년 단위 평균 변화
    st.subheader("📅 10년 단위 평균기온 변화")
    annual["10년대"] = (annual["연도"] // 10) * 10
    decade_df = annual.groupby("10년대")["연평균기온"].mean().reset_index()
    decade_df.columns = ["10년대", "평균기온"]
    decade_df["변화(전 대비)"] = decade_df["평균기온"].diff().round(2)
    decade_df["평균기온"] = decade_df["평균기온"].round(2)

    fig, ax = plt.subplots(figsize=(13, 4))
    colors_bar = [COLOR_BEFORE if y < cutoff else COLOR_AFTER for y in decade_df["10년대"]]
    bars = ax.bar(decade_df["10년대"].astype(str) + "s", decade_df["평균기온"], color=colors_bar, edgecolor="white")
    for bar, val in zip(bars, decade_df["평균기온"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{val:.1f}", ha="center", va="bottom", fontsize=8)
    ax.axvline(x=decade_df[decade_df["10년대"] == (cutoff // 10) * 10].index[0] - 0.5
               if not decade_df[decade_df["10년대"] == (cutoff // 10) * 10].empty else 0,
               color="gray", linestyle="--", linewidth=1.5)
    ax.set_ylabel("평균기온 (°C)"); ax.set_title("10년 단위 평균기온")
    ax.grid(axis="y", alpha=0.3); plt.xticks(rotation=45)
    st.pyplot(fig, use_container_width=True)

    st.dataframe(decade_df, hide_index=True, use_container_width=True)

    st.markdown("""
    ---
    **📌 분석 해석 가이드**
    - p-값 < 0.05: 이전/이후 기온 차이가 통계적으로 유의함 (우연이 아님)
    - 상승 속도(°C/10년): 10년당 기온 상승 정도 — 클수록 빠른 상승
    - 이후 기간의 상승 속도가 이전보다 크다면 **가속화된 온난화** 가설 지지
    """)

# ── 푸터 ─────────────────────────────────────────────────────
st.markdown("---")
st.caption("데이터: 기상청 서울(지점 108) 일별 기온 관측값 | 분석: Streamlit + SciPy")
