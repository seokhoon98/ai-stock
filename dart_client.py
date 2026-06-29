"""
재무 데이터 클라이언트 — Yahoo Finance(yfinance) 기반.

yfinance는 KRX 상장 종목을 .KS(코스피) / .KQ(코스닥) 심볼로 지원한다.
제공 데이터:
  - 연간 손익계산서 / 재무상태표 (다년도 추이)
  - 현재 밸류에이션 지표: PER, PBR, ROE, ROIC, 영업이익률, 부채비율, 배당수익률
"""

import pandas as pd

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False


class DartApiError(Exception):
    pass


def _to_yf_symbol(stock_code: str) -> list:
    code = stock_code.strip().zfill(6)
    return [f"{code}.KS", f"{code}.KQ"]


def _safe(val) -> float | None:
    try:
        v = float(val)
        return None if pd.isna(v) else v
    except (TypeError, ValueError):
        return None


def _get_ticker(stock_code: str):
    """종목코드로 유효한 yfinance Ticker 객체를 반환한다."""
    for sym in _to_yf_symbol(stock_code):
        try:
            t = yf.Ticker(sym)
            info = t.info
            if info.get("regularMarketPrice") or info.get("currentPrice") or info.get("sharesOutstanding"):
                return t
        except Exception:
            continue
    raise DartApiError(f"종목코드 {stock_code}를 Yahoo Finance에서 찾을 수 없습니다. (.KS/.KQ 모두 실패)")


def _get_row(df: pd.DataFrame, candidates: list):
    if df is None or df.empty:
        return None
    for name in candidates:
        if name in df.index:
            return df.loc[name]
    return None


# DART 호환용 더미
def fetch_corp_code_map(api_key: str) -> dict:
    return {}


def get_financial_summary(api_key: str, stock_code: str, corp_code_map: dict, years: list) -> list:
    """연간 손익계산서 + 재무상태표 데이터 반환."""
    if not YFINANCE_AVAILABLE:
        raise DartApiError("yfinance 패키지가 설치되지 않았습니다.")

    ticker = _get_ticker(stock_code)

    try:
        inc = ticker.financials
        bal = ticker.balance_sheet
    except Exception as e:
        raise DartApiError(f"Yahoo Finance 재무 데이터 조회 실패: {e}")

    rows = []
    for year in sorted(years, reverse=True):
        inc_col = next((c for c in (inc.columns if inc is not None and not inc.empty else []) if hasattr(c, 'year') and c.year == year), None)
        bal_col = next((c for c in (bal.columns if bal is not None and not bal.empty else []) if hasattr(c, 'year') and c.year == year), None)

        if inc_col is None:
            continue

        def iv(candidates):
            row = _get_row(inc, candidates)
            return _safe(row.get(inc_col)) if row is not None else None

        def bv(candidates):
            if bal_col is None:
                return None
            row = _get_row(bal, candidates)
            return _safe(row.get(bal_col)) if row is not None else None

        revenue   = iv(["Total Revenue", "Revenue"])
        op_profit = iv(["Operating Income", "Ebit"])
        net_income = iv(["Net Income", "Net Income Common Stockholders"])
        equity     = bv(["Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"])
        liabilities = bv(["Total Liabilities Net Minority Interest", "Total Liabilities"])
        total_debt  = bv(["Total Debt", "Long Term Debt And Capital Lease Obligation"])
        cash        = bv(["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"])

        op_margin  = (op_profit / revenue * 100) if (op_profit and revenue) else None
        debt_ratio = (liabilities / equity * 100) if (liabilities and equity) else None
        roe        = (net_income / equity * 100) if (net_income and equity) else None

        # ROIC = NOPAT / Invested Capital
        # NOPAT ≈ Operating Income × (1 - 법인세율 25% 가정)
        # Invested Capital = 자본 + 총부채 - 현금
        nopat = op_profit * 0.75 if op_profit else None
        invested_capital = None
        if equity and total_debt is not None:
            ic = equity + total_debt - (cash or 0)
            invested_capital = ic if ic > 0 else None
        roic = (nopat / invested_capital * 100) if (nopat and invested_capital) else None

        rows.append({
            "year": year,
            "revenue": revenue,
            "operating_profit": op_profit,
            "net_income": net_income,
            "op_margin": op_margin,
            "debt_ratio": debt_ratio,
            "roe": roe,
            "roic": roic,
        })

    return rows


def get_valuation_metrics(stock_code: str) -> dict:
    """
    현재 시점 밸류에이션 지표를 반환한다.
    반환 키: per, forward_per, pbr, roe, roic, op_margin, debt_ratio, dividend_yield, eps
    """
    if not YFINANCE_AVAILABLE:
        raise DartApiError("yfinance 패키지가 설치되지 않았습니다.")

    ticker = _get_ticker(stock_code)
    info = ticker.info

    def pct(key):
        v = _safe(info.get(key))
        return v * 100 if v is not None else None

    per           = _safe(info.get("trailingPE"))
    forward_per   = _safe(info.get("forwardPE"))
    pbr           = _safe(info.get("priceToBook"))
    roe           = pct("returnOnEquity")
    op_margin     = pct("operatingMargins")
    dividend_yield = pct("dividendYield")
    eps           = _safe(info.get("trailingEps"))

    # 부채비율: debtToEquity는 이미 % 단위로 제공됨 (100배 값)
    debt_ratio = _safe(info.get("debtToEquity"))

    # ROIC: yfinance에 직접 제공 없으므로 재무제표에서 계산
    roic = None
    try:
        inc = ticker.financials
        bal = ticker.balance_sheet
        if inc is not None and not inc.empty and bal is not None and not bal.empty:
            latest_inc = inc.columns[0]
            latest_bal = bal.columns[0]

            def iv(candidates):
                row = _get_row(inc, candidates)
                return _safe(row.get(latest_inc)) if row is not None else None

            def bv(candidates):
                row = _get_row(bal, candidates)
                return _safe(row.get(latest_bal)) if row is not None else None

            op_profit  = iv(["Operating Income", "Ebit"])
            equity     = bv(["Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"])
            total_debt = bv(["Total Debt", "Long Term Debt And Capital Lease Obligation"])
            cash       = bv(["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"])

            if op_profit and equity:
                nopat = op_profit * 0.75
                ic = equity + (total_debt or 0) - (cash or 0)
                if ic > 0:
                    roic = nopat / ic * 100
    except Exception:
        pass

    return {
        "per": per,
        "forward_per": forward_per,
        "pbr": pbr,
        "roe": roe,
        "roic": roic,
        "op_margin": op_margin,
        "debt_ratio": debt_ratio,
        "dividend_yield": dividend_yield,
        "eps": eps,
    }
