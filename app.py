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
from dart_client import fetch_corp_code_map, get_financial_summary, DartApiError
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


@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def load_corp_code_map(dart_api_key):
    return fetch_corp_code_map(dart_api_key)


def kis_ready(cfg):
    return bool(cfg["kis_app_key"] and cfg["kis_app_secret"] and cfg["kis_account_no"])


# ----------------------------------------------------------------------
# 섹션 1: 자산 현황
# ----------------------------------------------------------------------
def section_account_summary(cfg, summary):
    st.subheader("💰 자산 현황")
    if summary is None:
        st.info("KIS API 자격증명을 입력하면 자산 현황이 표시됩니다.")
        return

    cols = st.columns(4)
    total_eval = summary.get("tot_evlu_amt", "0")
    profit_amt = summary.get("evlu_pfls_smtl_amt", "0")
    purchase_amt = summary.get("pchs_amt_smtl_amt", "0")
    try:
        profit_rate = float(profit_amt) / float(purchase_amt) * 100 if float(purchase_amt) else 0.0
    except (ValueError, ZeroDivisionError):
        profit_rate = 0.0

    cols[0].metric("총평가금액", f"{int(float(total_eval)):,}원")
    cols[1].metric("매입금액", f"{int(float(purchase_amt)):,}원")
    cols[2].metric("평가손익", f"{int(float(profit_amt)):,}원")
    cols[3].metric("평가수익률", f"{profit_rate:.2f}%")


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
        df = pd.DataFrame(candidates)
        fmt = {"현재가": "{:,.0f}"} if "현재가" in df.columns and df["현재가"].notna().any() else {}
        st.dataframe(df.style.format(fmt), use_container_width=True, hide_index=True)
        st.caption(AI_DISCLAIMER)
        if st.button("이 종목들로 워치리스트 교체", key="apply_ai_watchlist"):
            new_codes = ",".join(
                c["종목코드"] for c in candidates if c.get("종목코드") and c["종목코드"] != "-"
            )
            if new_codes:
                st.session_state["watchlist_raw"] = new_codes
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

    rows = []
    errors = []
    for symbol in cfg["watchlist"]:
        try:
            price_data = load_price(
                cfg["kis_app_key"], cfg["kis_app_secret"], cfg["kis_account_no"],
                cfg["kis_is_virtual"], symbol,
            )
            rows.append({
                "종목코드": symbol,
                "종목명": price_data.get("hts_kor_isnm", "-"),
                "현재가": float(price_data.get("stck_prpr", "0") or 0),
                "전일대비": float(price_data.get("prdy_vrss", "0") or 0),
                "등락률(%)": float(price_data.get("prdy_ctrt", "0") or 0),
            })
        except KISApiError as e:
            errors.append(f"{symbol}: {e}")

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(
            df.style.format({"현재가": "{:,.0f}", "전일대비": "{:,.0f}", "등락률(%)": "{:.2f}"}),
            use_container_width=True, hide_index=True,
        )
    for err in errors:
        st.warning(err)

    return rows


# ----------------------------------------------------------------------
# 섹션 4: 핵심 재무 분석 (DART + Gemini)
# 보유 종목을 기본으로, 워치리스트를 추가 선택지로 제공
# ----------------------------------------------------------------------
def section_financial_analysis(cfg, holdings, watchlist_rows):
    st.subheader("📑 핵심 재무 분석")
    if not cfg["dart_api_key"]:
        st.warning("DART API 키가 설정되지 않았습니다. (GitHub Secrets → DART_API_KEY)")
        return

    # 보유 종목 우선, 워치리스트 추가
    options = {}
    for h in (holdings or []):
        name = h.get("prdt_name", "-")
        code = h.get("pdno", "")
        if code:
            options[f"[보유] {name} ({code})"] = code
    for r in (watchlist_rows or []):
        code = r["종목코드"]
        if code not in options.values():
            options[f"[관심] {r['종목명']} ({code})"] = code

    if not options:
        manual_code = st.text_input("종목코드 직접 입력 (예: 005930)", key="manual_fin_code")
        if manual_code:
            options[manual_code] = manual_code
        else:
            st.write("분석할 종목이 없습니다. KIS 자격증명 입력 또는 종목코드를 직접 입력해주세요.")
            return

    label = st.selectbox("분석 대상 종목", list(options.keys()), key="fin_stock_select")
    symbol = options[label]
    years = st.multiselect("조회 연도", [2025, 2024, 2023, 2022], default=[2025, 2024, 2023], key="fin_years")

    if st.button("재무 데이터 조회", key="fetch_financials"):
        try:
            with st.spinner("DART에서 재무 데이터를 가져오는 중..."):
                corp_map = load_corp_code_map(cfg["dart_api_key"])
                financials = get_financial_summary(cfg["dart_api_key"], symbol, corp_map, sorted(years))
        except DartApiError as e:
            st.error(str(e))
            return

        if not financials:
            st.warning("해당 종목/연도에 대한 재무 데이터를 찾지 못했습니다.")
            return

        st.session_state["fin_data"] = {"label": label, "financials": financials}

    fin = st.session_state.get("fin_data")
    if fin:
        financials = fin["financials"]
        df = pd.DataFrame(financials)[["year", "revenue", "operating_profit", "net_income", "debt_ratio", "roe"]]
        df.columns = ["연도", "매출액", "영업이익", "순이익", "부채비율(%)", "ROE(%)"]
        st.dataframe(
            df.style.format({
                "매출액": "{:,.0f}", "영업이익": "{:,.0f}", "순이익": "{:,.0f}",
                "부채비율(%)": "{:.1f}", "ROE(%)": "{:.1f}",
            }),
            use_container_width=True, hide_index=True,
        )

        if cfg["gemini_api_key"]:
            if st.button("AI 코멘트 생성", key="gen_commentary"):
                try:
                    with st.spinner("Gemini로 코멘트를 생성하는 중..."):
                        commentary = generate_financial_commentary(
                            cfg["gemini_api_key"], cfg["gemini_model"], fin["label"], financials,
                        )
                    st.session_state["fin_commentary"] = commentary
                except AIHelperError as e:
                    st.error(str(e))

        commentary = st.session_state.get("fin_commentary")
        if commentary:
            st.markdown(commentary)


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
