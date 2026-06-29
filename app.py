"""
주식 포트폴리오 대시보드 (Streamlit 버전)

Gemini API 키 / DART API 키는 GitHub Secrets(st.secrets)에서 읽는다.
KIS 자격증명(App Key / Secret / 계좌번호)은 사이드바에서 입력.

섹션: 자산 현황 / 보유 종목 / 투자 성향 기반 AI 워치리스트 추천 / 관심종목 워치리스트 /
     핵심 재무 분석 / 최근 리밸런싱 제안
"""

import time
from datetime import date

import pandas as pd
import plotly.express as px
import streamlit as st

from kis_client import KISClient, KISApiError
from dart_client import get_financial_summary, get_valuation_metrics, DartApiError
from ai_helper import (
    generate_financial_commentary,
    generate_rebalancing_table,
    generate_watchlist_suggestions,
    DIRECTION_OPTIONS,
    MARKET_CAP_OPTIONS,
    STYLE_OPTIONS,
    DISCLAIMER as AI_DISCLAIMER,
    AIHelperError,
)

st.set_page_config(page_title="주식 포트폴리오 대시보드", layout="wide")

DEFAULT_WATCHLIST = ""
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


# ----------------------------------------------------------------------
# Secrets 로딩 (GitHub Secrets → st.secrets)
# ----------------------------------------------------------------------
def load_secrets():
    """
    Streamlit secrets에서 API 키를 읽는다.
    Gemini 키는 GEMINI_API_KEY_1, GEMINI_API_KEY_2, ... 순서로 읽어 리스트로 반환한다.
    GEMINI_API_KEY 단일 키도 지원 (하위 호환).
    """
    gemini_keys = []
    dart_key = ""

    try:
        for i in range(1, 11):
            k = st.secrets.get(f"GEMINI_API_KEY_{i}", "")
            if k:
                gemini_keys.append(k)
        if not gemini_keys:
            k = st.secrets.get("GEMINI_API_KEY", "")
            if k:
                gemini_keys.append(k)
        dart_key = st.secrets.get("DART_API_KEY", "")
    except Exception:
        pass

    return gemini_keys, dart_key


# ----------------------------------------------------------------------
# 사이드바: KIS 자격증명만 입력 (Gemini/DART는 Secrets에서)
# ----------------------------------------------------------------------
def render_sidebar():
    st.sidebar.header("🔑 KIS API 자격증명")
    st.sidebar.caption("입력한 값은 이 브라우저 세션에만 보관되며 서버에 저장되지 않습니다.")

    with st.sidebar.expander("한국투자증권(KIS) API", expanded=True):
        kis_app_key = st.text_input("App Key", type="password", key="kis_app_key")
        kis_app_secret = st.text_input("App Secret", type="password", key="kis_app_secret")
        kis_account_no = st.text_input("계좌번호 (예: 12345678-01)", key="kis_account_no")
        kis_is_virtual = st.checkbox("모의투자 계좌", value=False, key="kis_is_virtual")

    with st.sidebar.expander("AI 모델 설정", expanded=False):
        gemini_model = st.text_input(
            "Gemini 모델명", value=DEFAULT_GEMINI_MODEL, key="gemini_model",
            help="모델명이 바뀌었거나 오류가 나면 최신 모델명으로 수정하세요.",
        )

    st.sidebar.header("⭐ 관심종목 워치리스트")
    watchlist_raw = st.sidebar.text_area(
        "종목코드 (쉼표로 구분)", value=DEFAULT_WATCHLIST, key="watchlist_raw",
        help="6자리 종목코드를 쉼표로 구분해서 입력하세요.",
    )

    gemini_keys, dart_key = load_secrets()

    return {
        "kis_app_key": kis_app_key,
        "kis_app_secret": kis_app_secret,
        "kis_account_no": kis_account_no,
        "kis_is_virtual": kis_is_virtual,
        "gemini_api_key": gemini_keys,   # list[str]
        "gemini_model": gemini_model,
        "dart_api_key": dart_key,
        "watchlist": [c.strip() for c in watchlist_raw.split(",") if c.strip()],
    }


# ----------------------------------------------------------------------
# KIS 데이터 로딩
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_kis_client(app_key, app_secret, account_no, is_virtual):
    return KISClient(app_key, app_secret, account_no, is_virtual)


@st.cache_data(ttl=60, show_spinner=False)
def load_balance(app_key, app_secret, account_no, is_virtual):
    client = get_kis_client(app_key, app_secret, account_no, is_virtual)
    holdings, summary = client.get_balance()
    return holdings, summary


@st.cache_data(ttl=60, show_spinner=False)
def load_price(app_key, app_secret, account_no, is_virtual, symbol):
    client = get_kis_client(app_key, app_secret, account_no, is_virtual)
    return client.get_price(symbol)



def kis_ready(cfg):
    return bool(cfg["kis_app_key"] and cfg["kis_app_secret"] and cfg["kis_account_no"])


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def _get_stock_name_yf(symbol: str) -> str:
    """KIS에서 종목명을 가져오지 못할 때 pykrx → yfinance 순으로 fallback."""
    # 1순위: pykrx (KRX 직접 조회, 한국어 정식 명칭)
    try:
        from pykrx import stock as pykrx_stock
        name = pykrx_stock.get_market_ticker_name(symbol)
        if name:
            return name
    except Exception:
        pass
    # 2순위: yfinance
    try:
        import yfinance as yf
        for suffix in [".KS", ".KQ"]:
            t = yf.Ticker(f"{symbol}{suffix}")
            name = t.info.get("shortName") or t.info.get("longName") or ""
            if name:
                return name
    except Exception:
        pass
    return symbol


# ----------------------------------------------------------------------
# 섹션 1: 자산 현황
# ----------------------------------------------------------------------
def section_account_summary(cfg, summary):
    st.subheader("💰 자산 현황")
    if summary is None:
        st.info("KIS API 자격증명을 입력하면 자산 현황이 표시됩니다.")
        return

    total_eval   = summary.get("tot_evlu_amt", "0")
    profit_amt   = summary.get("evlu_pfls_smtl_amt", "0")
    purchase_amt = summary.get("pchs_amt_smtl_amt", "0")
    cash         = summary.get("dnca_tot_amt", "0")  # 예수금(현금)
    try:
        profit_rate = float(profit_amt) / float(purchase_amt) * 100 if float(purchase_amt) else 0.0
    except (ValueError, ZeroDivisionError):
        profit_rate = 0.0

    cols = st.columns(5)
    cols[0].metric("총평가금액", f"{int(float(total_eval)):,}원")
    cols[1].metric("매입금액",   f"{int(float(purchase_amt)):,}원")
    cols[2].metric("평가손익",   f"{int(float(profit_amt)):,}원")
    cols[3].metric("평가수익률", f"{profit_rate:.2f}%")
    cols[4].metric("예수금(현금)", f"{int(float(cash)):,}원",
                   help="계좌 내 미투자 현금(예수금)입니다. 주식 매수에 바로 사용 가능한 금액입니다.")


# ----------------------------------------------------------------------
# 섹션 2: 보유 종목
# ----------------------------------------------------------------------
def section_holdings(holdings):
    st.subheader("📊 보유 종목")
    if holdings is None:
        st.info("KIS API 자격증명을 입력하면 보유 종목이 표시됩니다.")
        return None

    if not holdings:
        st.write("보유 중인 종목이 없습니다.")
        return None

    rows = []
    for h in holdings:
        rows.append({
            "종목명": h.get("prdt_name"),
            "종목코드": h.get("pdno"),
            "보유수량": int(h.get("hldg_qty", "0") or 0),
            "평균매입가": float(h.get("pchs_avg_pric", "0") or 0),
            "현재가": float(h.get("prpr", "0") or 0),
            "평가금액": float(h.get("evlu_amt", "0") or 0),
            "평가수익률(%)": float(h.get("evlu_pfls_rt", "0") or 0),
        })
    df = pd.DataFrame(rows)

    col1, col2 = st.columns([3, 2])
    with col1:
        st.dataframe(
            df.style.format({
                "평균매입가": "{:,.0f}", "현재가": "{:,.0f}",
                "평가금액": "{:,.0f}", "평가수익률(%)": "{:.2f}",
            }),
            use_container_width=True, hide_index=True,
        )
    with col2:
        fig = px.pie(df, names="종목명", values="평가금액", title="종목별 비중")
        st.plotly_chart(fig, use_container_width=True)

    return df


# ----------------------------------------------------------------------
# 섹션 3a: 투자 성향 기반 AI 워치리스트 추천
# ----------------------------------------------------------------------
def section_ai_watchlist_picker(cfg, holdings):
    st.subheader("🧭 투자 성향 기반 워치리스트 추천")
    if not cfg["gemini_api_key"]:
        st.warning("Gemini API 키가 설정되지 않았습니다. (GitHub Secrets → GEMINI_API_KEY)")
        return

    st.caption(
        "방향 / 시가총액 / 스타일 / 관심 테마를 답하면 보유 종목과 섹터가 겹치지 않게 "
        "신규 관심종목을 추천합니다."
    )

    with st.form("watchlist_picker_form"):
        col1, col2, col3 = st.columns(3)
        direction = col1.selectbox("방향", DIRECTION_OPTIONS, key="profile_direction")
        market_cap = col2.selectbox("시가총액 선호", MARKET_CAP_OPTIONS, key="profile_market_cap")
        style = col3.selectbox("투자 스타일", STYLE_OPTIONS, key="profile_style")
        themes = st.text_input(
            "관심 섹터/테마 (선택, 쉼표로 구분)", key="profile_themes",
            placeholder="예: 반도체, AI인프라, 바이오 (비워두면 시장 주도 테마 중 추천)",
        )
        submitted = st.form_submit_button("종목 추천받기")

    if submitted:
        exclude_names = [h.get("prdt_name") for h in (holdings or []) if h.get("prdt_name")]
        candidates = None
        try:
            with st.spinner("Gemini로 종목을 추천받는 중..."):
                candidates = generate_watchlist_suggestions(
                    cfg["gemini_api_key"], cfg["gemini_model"],
                    direction, market_cap, style, themes, exclude_names=exclude_names,
                )
        except AIHelperError as e:
            st.error(str(e))

        verified, warnings = [], []
        if candidates:
            if kis_ready(cfg):
                for c in candidates:
                    symbol = str(c.get("symbol", "")).strip()
                    if not symbol:
                        continue
                    try:
                        price_data = load_price(
                            cfg["kis_app_key"], cfg["kis_app_secret"],
                            cfg["kis_account_no"], cfg["kis_is_virtual"], symbol,
                        )
                        verified.append({
                            "종목코드": symbol,
                            "종목명": price_data.get("hts_kor_isnm") or c.get("name", "-"),
                            "테마": c.get("theme", "-"),
                            "현재가": float(price_data.get("stck_prpr", "0") or 0),
                            "추천 이유": c.get("reason", "-"),
                        })
                    except KISApiError as e:
                        warnings.append(f"{c.get('name', symbol)}({symbol}): KIS 조회 실패로 제외 — {e}")
            else:
                for c in candidates:
                    verified.append({
                        "종목코드": c.get("symbol", "-"),
                        "종목명": c.get("name", "-"),
                        "테마": c.get("theme", "-"),
                        "현재가": None,
                        "추천 이유": c.get("reason", "-"),
                    })
                warnings.append("KIS 자격증명이 없어 종목코드 유효성을 검증하지 못했습니다.")

        st.session_state["ai_watchlist_candidates"] = verified
        for w in warnings:
            st.warning(w)

    candidates = st.session_state.get("ai_watchlist_candidates")
    if candidates:
        display_rows = []
        for c in candidates:
            code = c.get("종목코드", "-")
            name = c.get("종목명", "-")
            display_rows.append({
                "종목명": name,
                "종목코드": code,
                "테마": c.get("테마", "-"),
                "현재가": c.get("현재가"),
                "추천 이유": c.get("추천 이유", "-"),
                "🔗": f"https://finance.naver.com/item/main.naver?code={code}" if code != "-" else "",
            })
        df = pd.DataFrame(display_rows)
        st.dataframe(
            df,
            column_config={
                "현재가": st.column_config.NumberColumn("현재가", format="%d"),
                "🔗": st.column_config.LinkColumn("🔗", display_text="🔗"),
            },
            use_container_width=True, hide_index=True,
        )
        st.caption(AI_DISCLAIMER)

        col1, col2 = st.columns(2)
        if col1.button("워치리스트에 추가", key="add_ai_watchlist"):
            new_codes = [c["종목코드"] for c in candidates if c.get("종목코드") and c["종목코드"] != "-"]
            existing = [c.strip() for c in st.session_state.get("watchlist_raw", "").split(",") if c.strip()]
            merged = existing + [c for c in new_codes if c not in existing]
            st.session_state["watchlist_pending"] = ",".join(merged)
            name_map = st.session_state.get("watchlist_name_map", {})
            name_map.update({c["종목코드"]: c["종목명"] for c in candidates if c.get("종목코드") != "-"})
            st.session_state["watchlist_name_map"] = name_map
            st.session_state["ai_watchlist_candidates"] = None
            st.rerun()
        if col2.button("워치리스트 교체", key="apply_ai_watchlist"):
            new_codes = ",".join(
                c["종목코드"] for c in candidates if c.get("종목코드") and c["종목코드"] != "-"
            )
            if new_codes:
                st.session_state["watchlist_pending"] = new_codes
                st.session_state["watchlist_name_map"] = {
                    c["종목코드"]: c["종목명"] for c in candidates if c.get("종목코드") != "-"
                }
                st.session_state["ai_watchlist_candidates"] = None
                st.rerun()


# ----------------------------------------------------------------------
# 섹션 3: 관심종목 워치리스트
# ----------------------------------------------------------------------
def section_watchlist(cfg):
    st.subheader("⭐ 관심종목 워치리스트")
    if not kis_ready(cfg):
        st.info("KIS API 자격증명을 입력하면 관심종목 시세가 표시됩니다.")
        return []

    if not cfg["watchlist"]:
        st.write("사이드바에서 관심종목 코드를 입력해주세요.")
        return []

    name_map = st.session_state.get("watchlist_name_map", {})
    rows = []
    errors = []
    for symbol in cfg["watchlist"]:
        try:
            price_data = load_price(
                cfg["kis_app_key"], cfg["kis_app_secret"], cfg["kis_account_no"],
                cfg["kis_is_virtual"], symbol,
            )
            name = (price_data.get("hts_kor_isnm") or "").strip()
            # KIS에 없으면 세션 저장 이름 → yfinance 순으로 fallback
            if not name:
                name = name_map.get(symbol, "")
            if not name:
                name = _get_stock_name_yf(symbol)
            rows.append({
                "종목코드": symbol,
                "종목명": name,
                "현재가": float(price_data.get("stck_prpr", "0") or 0),
                "전일대비": float(price_data.get("prdy_vrss", "0") or 0),
                "등락률(%)": float(price_data.get("prdy_ctrt", "0") or 0),
            })
        except KISApiError as e:
            errors.append(f"{symbol}: {e}")

    if rows:
        display = []
        for r in rows:
            code = r["종목코드"]
            name = r["종목명"]
            display.append({
                "종목명": name,
                "종목코드": code,
                "현재가": r["현재가"],
                "전일대비": r["전일대비"],
                "등락률(%)": r["등락률(%)"],
                "🔗": f"https://finance.naver.com/item/main.naver?code={code}",
            })
        df = pd.DataFrame(display)
        st.dataframe(
            df,
            column_config={
                "현재가": st.column_config.NumberColumn("현재가", format="%d"),
                "전일대비": st.column_config.NumberColumn("전일대비", format="%d"),
                "등락률(%)": st.column_config.NumberColumn("등락률(%)", format="%.2f"),
                "🔗": st.column_config.LinkColumn("🔗", display_text="🔗"),
            },
            use_container_width=True, hide_index=True,
        )
    for err in errors:
        st.warning(err)

    return rows


# ----------------------------------------------------------------------
# 섹션 4: 핵심 재무 분석 (DART + Gemini) — 종목별 탭
# ----------------------------------------------------------------------
def _render_fin_tab(cfg, label, symbol, tab_key):
    """단일 종목 재무 분석 탭 내용."""
    naver_url = f"https://finance.naver.com/item/main.naver?code={symbol}"
    st.link_button(f"{label} ↗ 네이버 금융", naver_url)

    years = st.multiselect("조회 연도", [2025, 2024, 2023, 2022], default=[2025, 2024, 2023], key=f"fin_years_{tab_key}")

    if st.button("재무 데이터 조회", key=f"fetch_fin_{tab_key}"):
        try:
            with st.spinner("Yahoo Finance에서 재무 데이터를 가져오는 중..."):
                financials = get_financial_summary("", symbol, {}, sorted(years))
                valuation  = get_valuation_metrics(symbol)
        except DartApiError as e:
            st.error(str(e))
            return

        if not financials:
            st.warning("해당 종목/연도에 대한 재무 데이터를 찾지 못했습니다.")
            return

        fin_store = st.session_state.get("fin_data", {})
        fin_store[tab_key] = {"label": label, "financials": financials, "valuation": valuation}
        st.session_state["fin_data"] = fin_store

    fin_store = st.session_state.get("fin_data", {})
    fin = fin_store.get(tab_key)
    if fin:
        financials = fin["financials"]
        valuation  = fin.get("valuation", {})

        def _fmt(v, suffix="", decimals=2):
            return f"{v:.{decimals}f}{suffix}" if v is not None else "-"

        # ── 현재 밸류에이션 지표 ──
        st.markdown("**📊 현재 밸류에이션**")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("PER (TTM)", _fmt(valuation.get("per"), "배", 1),
            help="주가수익비율 (Price/EPS). 주가가 순이익의 몇 배인지를 나타냅니다.\n"
                 "• 낮을수록 저평가 가능성 (단, 업종 평균과 비교 필요)\n"
                 "• 10~15배: 일반적 수준 / 20배 이상: 고성장 기대 반영\n"
                 "• 적자 기업은 PER 계산 불가 (-)")
        c2.metric("선행 PER", _fmt(valuation.get("forward_per"), "배", 1),
            help="향후 12개월 예상 이익 기준 PER입니다.\n"
                 "• TTM PER보다 낮으면 이익 성장이 기대됨\n"
                 "• 애널리스트 추정치 기반이므로 실제와 다를 수 있음")
        c3.metric("PBR", _fmt(valuation.get("pbr"), "배", 2),
            help="주가순자산비율 (Price/Book Value). 주가가 순자산의 몇 배인지를 나타냅니다.\n"
                 "• 1배 미만: 청산가치보다 싸게 거래 (저평가 신호일 수 있음)\n"
                 "• 높을수록 시장이 높은 프리미엄을 부여\n"
                 "• 금융주·제조업에서 특히 유용한 지표")
        c4.metric("EPS", _fmt(valuation.get("eps"), "원", 0),
            help="주당순이익 (Earnings Per Share). 주식 1주당 벌어들인 순이익입니다.\n"
                 "• 높을수록 수익성이 좋음\n"
                 "• PER 계산의 기준값 (주가 ÷ EPS = PER)")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("ROE", _fmt(valuation.get("roe"), "%", 1),
            help="자기자본이익률 (Return on Equity). 자본 대비 순이익 비율입니다.\n"
                 "• 10% 이상: 양호 / 15% 이상: 우수\n"
                 "• 높을수록 주주 자본을 효율적으로 활용\n"
                 "• 단, 부채를 많이 쓸수록 ROE가 높아질 수 있어 부채비율과 함께 봐야 함")
        c6.metric("ROIC", _fmt(valuation.get("roic"), "%", 1),
            help="투하자본이익률 (Return on Invested Capital). 실제 사업에 투입한 자본 대비 세후 영업이익 비율입니다.\n"
                 "• WACC(자본비용)보다 높으면 가치 창출 기업\n"
                 "• 10% 이상: 양호 / ROE보다 신뢰도 높은 수익성 지표\n"
                 "• 법인세율 25% 가정하여 계산")
        c7.metric("영업이익률", _fmt(valuation.get("op_margin"), "%", 1),
            help="영업이익 ÷ 매출액 × 100. 매출에서 실제 영업으로 남긴 이익 비율입니다.\n"
                 "• 업종마다 기준이 다름 (IT/바이오: 높음 / 유통·제조: 낮음)\n"
                 "• 10% 이상: 일반적으로 양호\n"
                 "• 추세가 개선되는지가 중요")
        c8.metric("부채비율", _fmt(valuation.get("debt_ratio"), "%", 1),
            help="총부채 ÷ 자기자본 × 100. 자본 대비 부채의 크기입니다.\n"
                 "• 100% 이하: 안정적 / 200% 이상: 주의 필요\n"
                 "• 업종 특성상 금융·인프라주는 부채비율이 높은 편\n"
                 "• ROE와 함께 보면 레버리지 효과 파악 가능")

        c9, *_ = st.columns(4)
        c9.metric("배당수익률", _fmt(valuation.get("dividend_yield"), "%", 2),
            help="연간 배당금 ÷ 현재 주가 × 100. 주가 대비 배당 수익 비율입니다.\n"
                 "• 3% 이상: 배당주로서 매력적\n"
                 "• 너무 높으면 주가 하락 또는 배당 유지 불가 가능성 점검 필요\n"
                 "• 무배당 종목은 '-'로 표시")

        st.divider()

        # ── 연간 재무 추이 ──
        st.markdown("**📈 연간 재무 추이**")
        cols_map = {
            "year": "연도", "revenue": "매출액", "operating_profit": "영업이익",
            "net_income": "순이익", "op_margin": "영업이익률(%)",
            "debt_ratio": "부채비율(%)", "roe": "ROE(%)", "roic": "ROIC(%)",
        }
        df_raw = pd.DataFrame(financials)
        avail = [c for c in cols_map if c in df_raw.columns]
        df = df_raw[avail].rename(columns=cols_map)

        col_help = {
            "매출액":       ("매출액 (원)", "%d", "기업이 영업활동으로 벌어들인 총 수익입니다.\n• 꾸준히 증가하면 성장하는 기업\n• 급격한 감소는 사업 위축 신호"),
            "영업이익":     ("영업이익 (원)", "%d", "매출에서 영업비용을 뺀 이익. 본업의 수익성을 나타냅니다.\n• 매출이 늘어도 영업이익이 줄면 비용 증가 문제\n• 영업이익이 마이너스면 본업에서 손실"),
            "순이익":       ("순이익 (원)", "%d", "영업이익에서 이자·세금·기타 비용까지 모두 뺀 최종 이익입니다.\n• EPS·ROE 계산의 기준\n• 일시적 비용으로 낮아질 수 있으므로 추세로 판단"),
            "영업이익률(%)":("영업이익률 (%)", "%.1f", "영업이익 ÷ 매출액 × 100. 매출 중 실제 남는 이익 비율입니다.\n• 업종 평균 대비 높으면 경쟁우위 보유\n• 추세 개선 여부가 핵심"),
            "부채비율(%)":  ("부채비율 (%)", "%.1f", "총부채 ÷ 자기자본 × 100. 재무 건전성 지표입니다.\n• 100% 이하: 안정적 / 200% 이상: 주의\n• 감소 추세면 재무구조 개선 중"),
            "ROE(%)":       ("ROE (%)", "%.1f", "순이익 ÷ 자기자본 × 100. 자본 대비 수익 창출 능력입니다.\n• 10% 이상: 양호 / 15% 이상: 우수\n• 부채비율과 함께 확인 필요"),
            "ROIC(%)":      ("ROIC (%)", "%.1f", "세후영업이익 ÷ 투하자본 × 100. 실제 사업 투자 효율입니다.\n• WACC보다 높으면 가치 창출 기업\n• ROE보다 부채 영향을 덜 받아 신뢰도 높음"),
        }

        col_config = {"연도": st.column_config.NumberColumn("연도", format="%d")}
        for col_name, (label, fmt_str, help_text) in col_help.items():
            if col_name in df.columns:
                col_config[col_name] = st.column_config.NumberColumn(col_name, format=fmt_str, help=help_text)

        st.dataframe(df, column_config=col_config, use_container_width=True, hide_index=True)

        if cfg["gemini_api_key"]:
            if st.button("AI 코멘트 생성", key=f"gen_commentary_{tab_key}"):
                try:
                    with st.spinner("Gemini로 애널리스트 리포트를 생성하는 중..."):
                        commentary = generate_financial_commentary(
                            cfg["gemini_api_key"], cfg["gemini_model"], fin["label"], financials,
                            valuation=valuation,
                        )
                    comm_store = st.session_state.get("fin_commentary", {})
                    comm_store[tab_key] = commentary
                    st.session_state["fin_commentary"] = comm_store
                except AIHelperError as e:
                    st.error(str(e))

        comm_store = st.session_state.get("fin_commentary", {})
        if comm_store.get(tab_key):
            st.markdown(comm_store[tab_key])


def section_financial_analysis(cfg, holdings, watchlist_rows):
    st.subheader("📑 핵심 재무 분석")

    # 보유 종목 우선, 워치리스트 추가
    options = {}
    for h in (holdings or []):
        name = h.get("prdt_name", "-")
        code = h.get("pdno", "")
        if code:
            options[name] = code
    for r in (watchlist_rows or []):
        code = r["종목코드"]
        name = r["종목명"]
        if code not in options.values():
            options[name] = code

    if not options:
        manual_code = st.text_input("종목코드 직접 입력 (예: 005930)", key="manual_fin_code")
        if manual_code:
            options[manual_code] = manual_code
        else:
            st.write("분석할 종목이 없습니다. KIS 자격증명 입력 또는 종목코드를 직접 입력해주세요.")
            return

    tab_labels = list(options.keys())
    tabs = st.tabs(tab_labels)
    for tab, name in zip(tabs, tab_labels):
        with tab:
            _render_fin_tab(cfg, name, options[name], tab_key=options[name])


# ----------------------------------------------------------------------
# 섹션 5: 최근 리밸런싱 제안 — 표 형식 + 실행 상태 추적
# ----------------------------------------------------------------------
def _status_badge(status: str) -> str:
    if status == "실행됨":
        return "🟢 실행됨"
    return "⬜ 미실행"


def section_rebalancing(cfg, holdings_df, holdings, summary, watchlist_rows):
    st.subheader("🔄 최근 리밸런싱 제안")
    if not cfg["gemini_api_key"]:
        st.warning("Gemini API 키가 설정되지 않았습니다. (GitHub Secrets → GEMINI_API_KEY)")
        return

    if holdings_df is None or summary is None:
        st.write("보유 종목/계좌 데이터가 필요합니다. KIS 자격증명을 먼저 입력해주세요.")
        return

    if st.button("AI 리밸런싱 제안 생성", key="gen_rebalancing"):
        holdings_payload = [
            {
                "name": row["종목명"], "symbol": row["종목코드"],
                "qty": row["보유수량"], "eval_amount": row["평가금액"],
                "profit_rate": row["평가수익률(%)"],
            }
            for _, row in holdings_df.iterrows()
        ]
        watchlist_payload = [
            {"name": r["종목명"], "symbol": r["종목코드"], "price": r["현재가"]}
            for r in (watchlist_rows or [])
        ]

        try:
            total_eval = summary.get("tot_evlu_amt", "0")
            profit_amt = summary.get("evlu_pfls_smtl_amt", "0")
            purchase_amt = summary.get("pchs_amt_smtl_amt", "0")
            profit_rate = float(profit_amt) / float(purchase_amt) * 100 if float(purchase_amt) else 0.0
            account_summary = {"total_eval_amount": total_eval, "total_profit_rate": round(profit_rate, 2)}

            with st.spinner("Gemini로 리밸런싱 제안을 생성하는 중..."):
                proposals = generate_rebalancing_table(
                    cfg["gemini_api_key"], cfg["gemini_model"],
                    holdings_payload, watchlist_payload, account_summary,
                )

            today = date.today().strftime("%y-%m-%d")
            st.session_state["rebalancing"] = {
                "date": today,
                "rows": [
                    {
                        "종목": p.get("name", "-"),
                        "종목코드": p.get("symbol", "-"),
                        "액션": p.get("action", "-"),
                        "수량": p.get("qty", 0),
                        "사유": p.get("reason", "-"),
                        "상태": "미실행",
                    }
                    for p in proposals
                ],
            }
        except (AIHelperError, Exception) as e:
            st.error(str(e))

    rb = st.session_state.get("rebalancing")
    if not rb:
        return

    rows = rb["rows"]
    executed = sum(1 for r in rows if r["상태"] == "실행됨")
    total = len(rows)

    # 헤더 행: 제안일 + 실행 현황 뱃지
    col_title, col_badge = st.columns([4, 1])
    with col_title:
        st.caption(f"제안일: {rb['date']}")
    with col_badge:
        if executed == 0:
            st.markdown("**미실행**")
        elif executed == total:
            st.markdown("**✅ 전체 실행**")
        else:
            st.markdown(f"**일부 실행 ({executed}/{total})**")

    # 표 렌더링
    header_cols = st.columns([2, 1.2, 0.8, 3, 1.2, 1])
    for col, label in zip(header_cols, ["종목", "액션", "수량", "사유", "상태", "토글"]):
        col.markdown(f"**{label}**")
    st.divider()

    for i, row in enumerate(rows):
        cols = st.columns([2, 1.2, 0.8, 3, 1.2, 1])
        action_color = "🔴" if "매도" in row["액션"] else "🔵"
        cols[0].write(row["종목"])
        cols[1].write(f"{action_color} {row['액션']}")
        cols[2].write(f"+{row['수량']}주" if "매수" in row["액션"] else f"-{row['수량']}주")
        cols[3].write(row["사유"])
        cols[4].write(_status_badge(row["상태"]))
        if cols[5].button(
            "실행" if row["상태"] == "미실행" else "취소",
            key=f"rb_toggle_{i}",
        ):
            rows[i]["상태"] = "실행됨" if rows[i]["상태"] == "미실행" else "미실행"
            st.rerun()

    st.divider()
    st.caption(
        f"※ {rb['date']} 제안 — 이 섹션은 정적 스냅샷이라 매매 체결을 자동 감지하지 않으며, "
        "확인 시 수동 갱신 필요합니다."
    )
    st.caption(AI_DISCLAIMER)

    if st.button("제안 초기화", key="reset_rebalancing"):
        st.session_state["rebalancing"] = None
        st.rerun()


# ----------------------------------------------------------------------
# 메인
# ----------------------------------------------------------------------
def main():
    st.title("📈 주식 포트폴리오 대시보드")
    st.caption("KIS / DART / Gemini API를 연결하는 개인용 대시보드입니다. 투자 자문이 아닙니다.")

    # 워치리스트 pending 값을 위젯 렌더링 전에 반영
    if "watchlist_pending" in st.session_state:
        st.session_state["watchlist_raw"] = st.session_state.pop("watchlist_pending")

    cfg = render_sidebar()

    holdings, summary = None, None
    if kis_ready(cfg):
        try:
            with st.spinner("KIS 계좌 정보를 불러오는 중..."):
                holdings, summary = load_balance(
                    cfg["kis_app_key"], cfg["kis_app_secret"],
                    cfg["kis_account_no"], cfg["kis_is_virtual"],
                )
        except KISApiError as e:
            st.error(f"KIS API 오류: {e}")
        except Exception as e:
            st.error(f"예상치 못한 오류: {e}")

    section_account_summary(cfg, summary)
    st.divider()
    holdings_df = section_holdings(holdings)
    st.divider()
    section_ai_watchlist_picker(cfg, holdings)
    st.divider()
    watchlist_rows = section_watchlist(cfg)
    st.divider()
    section_financial_analysis(cfg, holdings, watchlist_rows)
    st.divider()
    section_rebalancing(cfg, holdings_df, holdings, summary, watchlist_rows)


if __name__ == "__main__":
    main()
