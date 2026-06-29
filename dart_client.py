"""
재무 데이터 클라이언트 — Yahoo Finance(yfinance) 기반.

DART API는 Streamlit Cloud(미국 서버)에서 한국 금감원 서버에 접근이 차단되는 경우가 많아
yfinance로 대체했다. yfinance는 KRX 상장 종목을 .KS(코스피) / .KQ(코스닥) 심볼로 지원한다.

제공 데이터: 연간 손익계산서(매출액, 영업이익, 순이익) + 재무상태표(부채비율, ROE)
"""

import pandas as pd

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False


class DartApiError(Exception):
    pass


def _to_yf_symbol(stock_code: str) -> list[str]:
    """6자리 종목코드 → yfinance 심볼 후보 목록 (코스피 우선, 코스닥 차선)."""
    code = stock_code.strip().zfill(6)
    return [f"{code}.KS", f"{code}.KQ"]


def _safe_val(val) -> float | None:
    """pandas/numpy 값을 float으로 변환. NaN/None → None."""
    try:
        v = float(val)
        return None if pd.isna(v) else v
    except (TypeError, ValueError):
        return None


# DART API 호환용 더미 함수 (app.py에서 호출하지만 yfinance 방식에서는 불필요)
def fetch_corp_code_map(api_key: str) -> dict:
    return {}


def get_financial_summary(api_key: str, stock_code: str, corp_code_map: dict, years: list) -> list:
    """
    yfinance로 연간 재무 데이터를 가져온다.
    반환: [{"year", "revenue", "operating_profit", "net_income", "debt_ratio", "roe"}, ...]
    """
    if not YFINANCE_AVAILABLE:
        raise DartApiError("yfinance 패키지가 설치되지 않았습니다.")

    ticker = None
    for sym in _to_yf_symbol(stock_code):
        try:
            t = yf.Ticker(sym)
            info = t.info
            if info.get("regularMarketPrice") or info.get("currentPrice") or info.get("sharesOutstanding"):
                ticker = t
                break
        except Exception:
            continue

    if ticker is None:
        raise DartApiError(f"종목코드 {stock_code}를 Yahoo Finance에서 찾을 수 없습니다. (.KS/.KQ 모두 실패)")

    try:
        inc = ticker.financials          # 손익계산서 (연간), 열=날짜
        bal = ticker.balance_sheet       # 재무상태표 (연간)
    except Exception as e:
        raise DartApiError(f"Yahoo Finance 재무 데이터 조회 실패: {e}")

    def _get_row(df: pd.DataFrame, candidates: list):
        if df is None or df.empty:
            return None
        for name in candidates:
            if name in df.index:
                return df.loc[name]
        return None

    rows = []
    for year in sorted(years, reverse=True):
        # 해당 연도 열 찾기 (연도가 일치하는 열)
        col = None
        for c in (inc.columns if inc is not None and not inc.empty else []):
            if hasattr(c, 'year') and c.year == year:
                col = c
                break

        if col is None:
            continue

        def v(df, candidates):
            row = _get_row(df, candidates)
            if row is None:
                return None
            return _safe_val(row.get(col))

        revenue      = v(inc, ["Total Revenue", "Revenue"])
        op_profit    = v(inc, ["Operating Income", "Operating Revenue", "Ebit"])
        net_income   = v(inc, ["Net Income", "Net Income Common Stockholders"])

        # 재무상태표: 같은 연도 열 찾기
        bal_col = None
        for c in (bal.columns if bal is not None and not bal.empty else []):
            if hasattr(c, 'year') and c.year == year:
                bal_col = c
                break

        def bv(candidates):
            if bal_col is None:
                return None
            row = _get_row(bal, candidates)
            if row is None:
                return None
            return _safe_val(row.get(bal_col))

        equity      = bv(["Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"])
        liabilities = bv(["Total Liabilities Net Minority Interest", "Total Liabilities"])

        debt_ratio = (liabilities / equity * 100) if (liabilities and equity) else None
        roe        = (net_income / equity * 100) if (net_income and equity) else None

        rows.append({
            "year": year,
            "revenue": revenue,
            "operating_profit": op_profit,
            "net_income": net_income,
            "debt_ratio": debt_ratio,
            "roe": roe,
        })

    return rows
