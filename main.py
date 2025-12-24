import io
import unicodedata
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


# -----------------------------
# Page & Global Styling
# -----------------------------
st.set_page_config(page_title="🌱 극지식물 최적 EC 농도 연구", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
    font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
}
</style>
""",
    unsafe_allow_html=True,
)

PLOTLY_FONT = dict(family="Malgun Gothic, Apple SD Gothic Neo, Noto Sans KR, sans-serif")

SCHOOLS = ["송도고", "하늘고", "아라고", "동산고"]

# 학교별 EC 조건 (고정 정보이므로 하드코딩 OK)
EC_TARGET_BY_SCHOOL = {
    "송도고": 1.0,
    "하늘고": 2.0,  # 최적
    "아라고": 4.0,
    "동산고": 8.0,
}

SCHOOL_COLOR = {
    "송도고": "#1f77b4",
    "하늘고": "#2ca02c",  # 최적 강조용(초록)
    "아라고": "#ff7f0e",
    "동산고": "#d62728",
}

ENV_REQUIRED_COLS = ["time", "temperature", "humidity", "ph", "ec"]
GROWTH_REQUIRED_COLS = ["개체번호", "잎 수(장)", "지상부 길이(mm)", "지하부길이(mm)", "생중량(g)"]


# -----------------------------
# Unicode-safe helpers (NFC/NFD)
# -----------------------------
def norm_nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def norm_nfd(s: str) -> str:
    return unicodedata.normalize("NFD", s)


def unicode_equal(a: str, b: str) -> bool:
    """NFC/NFD 양방향 비교"""
    return (norm_nfc(a) == norm_nfc(b)) or (norm_nfd(a) == norm_nfd(b))


def find_file_by_exact_name(data_dir: Path, target_name: str) -> Optional[Path]:
    """
    pathlib.Path.iterdir()로 파일을 훑고,
    파일명은 NFC/NFD 양방향 normalize로 정확 일치 비교.
    (f-string 조합 금지 / glob-only 금지 대응)
    """
    if not data_dir.exists():
        return None
    for p in data_dir.iterdir():
        if p.is_file() and unicode_equal(p.name, target_name):
            return p
    return None


def find_env_csvs_by_school(data_dir: Path) -> Dict[str, Path]:
    """
    data_dir.iterdir() 기반으로 CSV들을 찾아 학교별 매핑.
    파일명/경로 f-string 조합 없이, 실제 존재 파일에서 '학교명' 포함 여부로 분류.
    """
    result: Dict[str, Path] = {}
    if not data_dir.exists():
        return result

    for p in data_dir.iterdir():
        if not p.is_file():
            continue
        # 확장자 비교도 NFC/NFD 안전하게 처리
        if norm_nfc(p.suffix.lower()) != ".csv":
            continue

        name_nfc = norm_nfc(p.name)
        # "환경데이터" 키워드가 있을 때만 후보로
        if "환경데이터" not in name_nfc:
            continue

        for school in SCHOOLS:
            if school in name_nfc:
                # 같은 학교가 여러 번 매칭되면 첫 번째만 사용
                result.setdefault(school, p)
                break

    return result


def read_csv_robust(path: Path) -> pd.DataFrame:
    """
    Streamlit Cloud에서도 안정적으로 읽도록 인코딩/파싱을 방어적으로 처리.
    """
    last_err = None
    for enc in ["utf-8-sig", "utf-8", "cp949"]:
        try:
            df = pd.read_csv(path, encoding=enc)
            return df
        except Exception as e:
            last_err = e
            continue
    raise last_err  # type: ignore


def ensure_columns(df: pd.DataFrame, required: List[str]) -> Tuple[bool, List[str]]:
    missing = [c for c in required if c not in df.columns]
    return (len(missing) == 0, missing)


def to_xlsx_bytes(df: pd.DataFrame, sheet_name: str = "data") -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    buffer.seek(0)
    return buffer.getvalue()


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


# -----------------------------
# Data Loaders (cached)
# -----------------------------
@st.cache_data(show_spinner=False)
def load_env_data(data_dir: Path) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Path]]:
    env_paths = find_env_csvs_by_school(data_dir)
    env_dfs: Dict[str, pd.DataFrame] = {}

    for school, path in env_paths.items():
        df = read_csv_robust(path).copy()

        ok, missing = ensure_columns(df, ENV_REQUIRED_COLS)
        if not ok:
            # 컬럼 불일치면 빈 DF로 처리(에러는 상위에서 안내)
            env_dfs[school] = pd.DataFrame()
            continue

        # time 파싱
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        # 수치형 변환
        for col in ["temperature", "humidity", "ph", "ec"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["time"]).sort_values("time")
        df["school"] = school
        env_dfs[school] = df

    return env_dfs, env_paths


@st.cache_data(show_spinner=False)
def load_growth_data(data_dir: Path) -> Tuple[pd.DataFrame, Optional[Path], List[str]]:
    """
    4개교_생육결과데이터.xlsx 파일을 찾아 모든 시트를 읽어 long-format으로 합침.
    - 시트명 하드코딩 금지: sheet_name=None으로 전부 읽기
    - 시트→학교 매핑은 "시트명에 학교명 포함"으로 유연하게 처리
    """
    # 정확 파일명 탐색(요구 구조)
    xlsx_path = find_file_by_exact_name(data_dir, "4개교_생육결과데이터.xlsx")
    if xlsx_path is None:
        # 그래도 폴더 내 .xlsx 중 유사 파일 찾기(백업)
        for p in data_dir.iterdir() if data_dir.exists() else []:
            if p.is_file() and norm_nfc(p.suffix.lower()) == ".xlsx":
                if "생육결과" in norm_nfc(p.name):
                    xlsx_path = p
                    break

    if xlsx_path is None:
        return pd.DataFrame(), None, []

    sheets = pd.read_excel(xlsx_path, sheet_name=None, engine="openpyxl")
    sheet_names = list(sheets.keys())

    frames = []
    for sheet_name, df in sheets.items():
        df = df.copy()
        # 학교명 추정: 시트명에 학교명 포함 시 매핑, 아니면 시트명을 학교로 사용
        school_guess = None
        sheet_nfc = norm_nfc(str(sheet_name))
        for school in SCHOOLS:
            if school in sheet_nfc:
                school_guess = school
                break
        if school_guess is None:
            school_guess = sheet_nfc

        ok, missing = ensure_columns(df, GROWTH_REQUIRED_COLS)
        if not ok:
            # 컬럼이 다르면 스킵하되, 전체가 비면 상위에서 에러 안내
            continue

        for col in ["잎 수(장)", "지상부 길이(mm)", "지하부길이(mm)", "생중량(g)"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["school"] = school_guess
        df["ec_target"] = EC_TARGET_BY_SCHOOL.get(school_guess, None)

        frames.append(df)

    if not frames:
        return pd.DataFrame(), xlsx_path, sheet_names

    long_df = pd.concat(frames, ignore_index=True)

    return long_df, xlsx_path, sheet_names


# -----------------------------
# UI
# -----------------------------
st.title("🌱 극지식물 최적 EC 농도 연구")

data_dir = Path(__file__).parent / "data"

with st.sidebar:
    st.header("옵션")
    school_choice = st.selectbox("학교 선택", ["전체"] + SCHOOLS, index=0)

with st.spinner("데이터를 불러오는 중..."):
    env_dfs, env_paths = load_env_data(data_dir)
    growth_df, growth_xlsx_path, growth_sheet_names = load_growth_data(data_dir)

# 데이터 유효성 체크
missing_env = [s for s in SCHOOLS if (s not in env_dfs) or env_dfs[s].empty]
if len(env_paths) == 0:
    st.error("환경 데이터(CSV) 파일을 data/ 폴더에서 찾지 못했습니다. 파일명(한글/정규화) 및 위치를 확인하세요.")
elif missing_env:
    st.warning(f"일부 학교의 환경 데이터가 비어있거나 컬럼이 맞지 않습니다: {', '.join(missing_env)}")

if growth_xlsx_path is None:
    st.error("생육 결과 XLSX 파일(4개교_생육결과데이터.xlsx)을 data/ 폴더에서 찾지 못했습니다.")
elif growth_df.empty:
    st.error("생육 결과 데이터는 읽었지만, 필수 컬럼이 맞는 시트를 찾지 못했습니다. 컬럼명을 확인하세요.")

tabs = st.tabs(["📖 실험 개요", "🌡️ 환경 데이터", "📊 생육 결과"])


# -----------------------------
# Tab 1: Overview
# -----------------------------
with tabs[0]:
    st.subheader("연구 배경 및 목적")
    st.write(
        """
본 연구는 **EC(전기전도도) 농도 조건**이 극지식물의 생육(생중량, 잎 수, 길이)에 미치는 영향을 비교하여,
학교별로 서로 다른 EC 조건에서 얻은 결과를 종합해 **최적 EC 농도(하늘고 EC 2.0)**를 도출하는 것을 목표로 합니다.
        """
    )

    # 학교별 EC 조건 표
    st.subheader("학교별 EC 조건")
    counts_by_school = {}
    if not growth_df.empty:
        counts_by_school = growth_df.groupby("school")["개체번호"].count().to_dict()

    table_rows = []
    for s in SCHOOLS:
        table_rows.append(
            {
                "학교명": s,
                "EC 목표": EC_TARGET_BY_SCHOOL.get(s),
                "개체수": int(counts_by_school.get(s, 0)),
                "색상": SCHOOL_COLOR.get(s, "#999999"),
            }
        )
    ec_table = pd.DataFrame(table_rows)
    st.dataframe(ec_table, use_container_width=True, hide_index=True)

    # 주요 지표 카드 4개
    total_n = int(growth_df["개체번호"].count()) if not growth_df.empty else 0

    # 환경 평균(전체 평균)
    all_env = pd.concat([df for df in env_dfs.values() if not df.empty], ignore_index=True) if env_dfs else pd.DataFrame()
    avg_temp = float(all_env["temperature"].mean()) if not all_env.empty else float("nan")
    avg_hum = float(all_env["humidity"].mean()) if not all_env.empty else float("nan")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("총 개체수", f"{total_n:,} 개")
    c2.metric("평균 온도", "-" if pd.isna(avg_temp) else f"{avg_temp:.2f} °C")
    c3.metric("평균 습도", "-" if pd.isna(avg_hum) else f"{avg_hum:.2f} %")
    c4.metric("최적 EC", "2.0 (하늘고)")


# -----------------------------
# Tab 2: Environment
# -----------------------------
with tabs[1]:
    st.subheader("학교별 환경 평균 비교")

    # 학교별 평균 집계
    env_summary_rows = []
    for s in SCHOOLS:
        df = env_dfs.get(s, pd.DataFrame())
        if df.empty:
            env_summary_rows.append(
                {"school": s, "temperature": None, "humidity": None, "ph": None, "ec_mean": None}
            )
            continue
        env_summary_rows.append(
            {
                "school": s,
                "temperature": df["temperature"].mean(),
                "humidity": df["humidity"].mean(),
                "ph": df["ph"].mean(),
                "ec_mean": df["ec"].mean(),
            }
        )
    env_summary = pd.DataFrame(env_summary_rows)
    env_summary["ec_target"] = env_summary["school"].map(EC_TARGET_BY_SCHOOL)

    # 2x2 서브플롯
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("평균 온도", "평균 습도", "평균 pH", "목표 EC vs 실측 EC(평균)"),
        horizontal_spacing=0.12,
        vertical_spacing=0.18,
    )

    # 평균 온도
    fig.add_trace(
        go.Bar(
            x=env_summary["school"],
            y=env_summary["temperature"],
            name="Avg Temp",
        ),
        row=1,
        col=1,
    )
    # 평균 습도
    fig.add_trace(
        go.Bar(
            x=env_summary["school"],
            y=env_summary["humidity"],
            name="Avg Humidity",
        ),
        row=1,
        col=2,
    )
    # 평균 pH
    fig.add_trace(
        go.Bar(
            x=env_summary["school"],
            y=env_summary["ph"],
            name="Avg pH",
        ),
        row=2,
        col=1,
    )
    # 목표 EC vs 실측 EC(이중 막대)
    fig.add_trace(
        go.Bar(
            x=env_summary["school"],
            y=env_summary["ec_target"],
            name="Target EC",
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Bar(
            x=env_summary["school"],
            y=env_summary["ec_mean"],
            name="Measured EC (Mean)",
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        barmode="group",
        height=650,
        margin=dict(l=30, r=30, t=70, b=30),
        font=PLOTLY_FONT,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("선택한 학교 시계열")

    def get_selected_env_df(choice: str) -> pd.DataFrame:
        if choice == "전체":
            return pd.concat([d for d in env_dfs.values() if not d.empty], ignore_index=True) if env_dfs else pd.DataFrame()
        return env_dfs.get(choice, pd.DataFrame()).copy()

    sel_env = get_selected_env_df(school_choice)

    if sel_env.empty:
        st.error("선택한 범위의 환경 데이터가 없습니다.")
    else:
        # 전체일 때는 school별 색상, 단일일 때는 단색
        color_map = SCHOOL_COLOR

        # 온도 변화
        fig_t = px.line(
            sel_env,
            x="time",
            y="temperature",
            color="school" if school_choice == "전체" else None,
            title="온도 변화",
        )
        fig_t.update_layout(font=PLOTLY_FONT, height=330, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_t, use_container_width=True)

        # 습도 변화
        fig_h = px.line(
            sel_env,
            x="time",
            y="humidity",
            color="school" if school_choice == "전체" else None,
            title="습도 변화",
        )
        fig_h.update_layout(font=PLOTLY_FONT, height=330, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_h, use_container_width=True)

        # EC 변화 + 목표 EC 수평선
        fig_e = px.line(
            sel_env,
            x="time",
            y="ec",
            color="school" if school_choice == "전체" else None,
            title="EC 변화 (목표 EC 수평선 포함)",
        )

        if school_choice == "전체":
            # 학교별 목표선 4개를 각각 추가(데이터 범위에 맞추어)
            x0 = sel_env["time"].min()
            x1 = sel_env["time"].max()
            for s in SCHOOLS:
                if s in sel_env["school"].unique():
                    target = EC_TARGET_BY_SCHOOL.get(s)
                    if target is not None and pd.notna(x0) and pd.notna(x1):
                        fig_e.add_shape(
                            type="line",
                            x0=x0,
                            x1=x1,
                            y0=target,
                            y1=target,
                            line=dict(dash="dash"),
                        )
        else:
            target = EC_TARGET_BY_SCHOOL.get(school_choice)
            if target is not None:
                x0 = sel_env["time"].min()
                x1 = sel_env["time"].max()
                if pd.notna(x0) and pd.notna(x1):
                    fig_e.add_shape(
                        type="line",
                        x0=x0,
                        x1=x1,
                        y0=target,
                        y1=target,
                        line=dict(dash="dash"),
                    )

        fig_e.update_layout(font=PLOTLY_FONT, height=360, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_e, use_container_width=True)

        with st.expander("원본 환경 데이터 보기 및 다운로드"):
            show_df = sel_env.copy()
            st.dataframe(show_df, use_container_width=True)

            # 다운로드는 '선택' 단위로 제공 (전체면 결합 CSV)
            st.download_button(
                label="CSV 다운로드",
                data=to_csv_bytes(show_df),
                file_name="환경데이터_선택범위.csv",
                mime="text/csv",
            )


# -----------------------------
# Tab 3: Growth
# -----------------------------
with tabs[2]:
    st.subheader("🥇 핵심 결과: EC별 평균 생중량")

    # 선택 학교 필터
    if growth_df.empty:
        st.error("생육 결과 데이터가 없습니다.")
    else:
        if school_choice == "전체":
            g = growth_df.copy()
        else:
            g = growth_df[growth_df["school"] == school_choice].copy()

        # EC 조건(학교별) 기준으로 평균 생중량 비교
        summary = (
            g.groupby(["school", "ec_target"], dropna=False)["생중량(g)"]
            .mean()
            .reset_index()
            .rename(columns={"생중량(g)": "avg_weight"})
        )

        # 전체 보기일 때는 4개교 비교 카드 형태
        if school_choice == "전체":
            # 최댓값
            best_row = summary.dropna(subset=["avg_weight"]).sort_values("avg_weight", ascending=False).head(1)
            best_text = "-"
            if not best_row.empty:
                best_school = best_row.iloc[0]["school"]
                best_ec = best_row.iloc[0]["ec_target"]
                best_w = best_row.iloc[0]["avg_weight"]
                best_text = f"{best_school} (EC {best_ec}) / {best_w:.3f} g"

            # 하늘고 강조(최적)
            sky_row = summary[summary["school"] == "하늘고"]
            sky_text = "-"
            if not sky_row.empty and pd.notna(sky_row.iloc[0]["avg_weight"]):
                sky_text = f"{sky_row.iloc[0]['avg_weight']:.3f} g"

            c1, c2 = st.columns(2)
            c1.metric("최대 평균 생중량", best_text)
            c2.metric("하늘고(EC 2.0) 평균 생중량", sky_text)
        else:
            # 단일 학교면 해당 학교 평균만 카드
            avg_w = float(g["생중량(g)"].mean()) if g["생중량(g)"].notna().any() else float("nan")
            target = EC_TARGET_BY_SCHOOL.get(school_choice)
            st.metric("평균 생중량", "-" if pd.isna(avg_w) else f"{avg_w:.3f} g", delta=f"EC 목표: {target}")

        st.divider()
        st.subheader("EC별 생육 비교 (2x2)")

        # 전체 기준: 학교(=EC조건)별 비교가 곧 EC별 비교
        base = growth_df.copy() if school_choice == "전체" else g.copy()

        agg = base.groupby("school").agg(
            avg_weight=("생중량(g)", "mean"),
            avg_leaves=("잎 수(장)", "mean"),
            avg_shoot=("지상부 길이(mm)", "mean"),
            n=("개체번호", "count"),
        ).reset_index()
        agg["ec_target"] = agg["school"].map(EC_TARGET_BY_SCHOOL)

        # 2x2 막대 그래프
        fig2 = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=("평균 생중량 (⭐ 가장 중요)", "평균 잎 수", "평균 지상부 길이", "개체수 비교"),
            horizontal_spacing=0.12,
            vertical_spacing=0.18,
        )

        # 평균 생중량(하늘고 강조는 주석/텍스트로 표시)
        fig2.add_trace(
            go.Bar(x=agg["school"], y=agg["avg_weight"], name="Avg Weight"),
            row=1, col=1
        )
        # 평균 잎 수
        fig2.add_trace(
            go.Bar(x=agg["school"], y=agg["avg_leaves"], name="Avg Leaves"),
            row=1, col=2
        )
        # 평균 지상부 길이
        fig2.add_trace(
            go.Bar(x=agg["school"], y=agg["avg_shoot"], name="Avg Shoot Length"),
            row=2, col=1
        )
        # 개체수
        fig2.add_trace(
            go.Bar(x=agg["school"], y=agg["n"], name="Count"),
            row=2, col=2
        )

        # 하늘고(EC 2.0) 텍스트 강조(그래프 위에)
        if "하늘고" in agg["school"].values:
            sky_val = agg.loc[agg["school"] == "하늘고", "avg_weight"].iloc[0]
            fig2.add_annotation(
                text="✅ 최적(하늘고 EC 2.0)",
                x="하늘고",
                y=sky_val if pd.notna(sky_val) else 0,
                xref="x1",
                yref="y1",
                showarrow=True,
                arrowhead=2,
            )

        fig2.update_layout(
            height=650,
            margin=dict(l=30, r=30, t=70, b=30),
            font=PLOTLY_FONT,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.divider()
        st.subheader("학교별 생중량 분포")

        dist_df = base.dropna(subset=["생중량(g)"]).copy()
        if dist_df.empty:
            st.error("생중량 분포를 그릴 데이터가 없습니다(생중량(g) 결측).")
        else:
            fig_box = px.box(
                dist_df,
                x="school",
                y="생중량(g)",
                points="outliers",
                title="생중량 분포 (Box Plot)",
            )
            fig_box.update_layout(font=PLOTLY_FONT, height=420, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_box, use_container_width=True)

        st.divider()
        st.subheader("상관관계 분석 (산점도 2개)")

        # 잎 수 vs 생중량
        sc1_df = base.dropna(subset=["잎 수(장)", "생중량(g)"]).copy()
        if sc1_df.empty:
            st.error("잎 수 vs 생중량 산점도를 그릴 데이터가 없습니다.")
        else:
            fig_sc1 = px.scatter(
                sc1_df,
                x="잎 수(장)",
                y="생중량(g)",
                color="school" if school_choice == "전체" else None,
                title="잎 수 vs 생중량",
                trendline=None,
            )
            fig_sc1.update_layout(font=PLOTLY_FONT, height=420, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_sc1, use_container_width=True)

        # 지상부 길이 vs 생중량
        sc2_df = base.dropna(subset=["지상부 길이(mm)", "생중량(g)"]).copy()
        if sc2_df.empty:
            st.error("지상부 길이 vs 생중량 산점도를 그릴 데이터가 없습니다.")
        else:
            fig_sc2 = px.scatter(
                sc2_df,
                x="지상부 길이(mm)",
                y="생중량(g)",
                color="school" if school_choice == "전체" else None,
                title="지상부 길이 vs 생중량",
                trendline=None,
            )
            fig_sc2.update_layout(font=PLOTLY_FONT, height=420, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_sc2, use_container_width=True)

        with st.expander("원본 생육 데이터 보기 및 XLSX 다운로드"):
            st.write(f"읽은 시트 목록: {', '.join([str(s) for s in growth_sheet_names]) if growth_sheet_names else '-'}")
            st.dataframe(base, use_container_width=True)

            xlsx_bytes = to_xlsx_bytes(base, sheet_name="growth")
            st.download_button(
                label="XLSX 다운로드",
                data=xlsx_bytes,
                file_name="생육데이터_선택범위.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
