"""
DART(전자공시시스템) OpenAPI 클라이언트 — 종목의 핵심 재무 데이터를 가져온다.

무료 API 키는 https://opendart.fss.no.kr 에서 발급받을 수 있다.
종목코드 -> corp_code 매핑은 DART가 제공하는 전체 corpCode.zip을 받아
세션 동안 메모리에 캐시해서 사용한다 (※ Streamlit의 st.cache_data로 캐시).

주의: DART 계정명(account_nm)은 회사마다 표기가 조금씩 다를 수 있어
(예: "매출액" vs "수익(매출액)") 후보 이름 목록으로 느슨하게 매칭한다.
완벽한 표준화는 아니므로 참고용으로만 사용할 것.
"""

import io
import zipfile
import xml.etree.ElementTree as ET

import requests

DART_BASE = "https://opendart.fss.or.kr/api"

# 계정명 후보 (느슨한 부분일치 매칭용)
ACCOUNT_ALIASES = {
    "revenue": ["매출액", "수익(매출액)", "영업수익"],
    "operating_profit": ["영업이익", "영업이익(손실)"],
    "net_income": ["당기순이익", "당기순이익(손실)", "분기순이익", "분기순이익(손실)"],
    "total_equity": ["자본총계"],
    "total_liabilities": ["부채총계"],
}


class DartApiError(Exception):
    pass


def fetch_corp_code_map(api_key: str) -> dict:
    """전체 상장사 corpCode.xml을 받아 {종목코드: corp_code} 딕셔너리로 반환."""
    resp = requests.get(f"{DART_BASE}/corpCode.xml", params={"crtfc_key": api_key}, timeout=20)
    resp.raise_for_status()
    if resp.headers.get("content-type", "").startswith("application/json"):
        # 키 오류 등은 zip이 아니라 json 에러로 내려온다
        raise DartApiError(f"DART corpCode 조회 실패: {resp.json()}")

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        xml_bytes = zf.read(zf.namelist()[0])

    root = ET.fromstring(xml_bytes)
    code_map = {}
    for item in root.findall("list"):
        stock_code = (item.findtext("stock_code") or "").strip()
        corp_code = (item.findtext("corp_code") or "").strip()
        if stock_code:
            code_map[stock_code] = corp_code
    return code_map


def _find_amount(rows: list, sj_div: str, aliases: list, amount_field: str):
    for row in rows:
        if row.get("sj_div") != sj_div:
            continue
        name = (row.get("account_nm") or "").replace(" ", "")
        if any(alias.replace(" ", "") in name for alias in aliases):
            raw = (row.get(amount_field) or "0").replace(",", "").strip()
            try:
                return int(raw)
            except ValueError:
                continue
    return None


def get_annual_financials(api_key: str, corp_code: str, year: int, fs_div: str = "CFS"):
    """
    사업보고서(연간, reprt_code=11011) 기준 핵심 계정 조회.
    fs_div: CFS(연결) 우선 시도, 데이터 없으면 OFS(별도)로 재시도.
    반환: {"year", "revenue", "operating_profit", "net_income",
           "total_equity", "total_liabilities", "debt_ratio", "roe"} 또는 None
    """
    params = {
        "crtfc_key": api_key,
        "corp_code": corp_code,
        "bsns_year": str(year),
        "reprt_code": "11011",
        "fs_div": fs_div,
    }
    resp = requests.get(f"{DART_BASE}/fnlttSinglAcntAll.json", params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    if data.get("status") != "000":
        if fs_div == "CFS":
            return get_annual_financials(api_key, corp_code, year, fs_div="OFS")
        return None

    rows = data.get("list", [])
    revenue = _find_amount(rows, "IS", ACCOUNT_ALIASES["revenue"], "thstrm_amount")
    op_profit = _find_amount(rows, "IS", ACCOUNT_ALIASES["operating_profit"], "thstrm_amount")
    net_income = _find_amount(rows, "IS", ACCOUNT_ALIASES["net_income"], "thstrm_amount")
    equity = _find_amount(rows, "BS", ACCOUNT_ALIASES["total_equity"], "thstrm_amount")
    liabilities = _find_amount(rows, "BS", ACCOUNT_ALIASES["total_liabilities"], "thstrm_amount")

    if revenue is None and op_profit is None:
        # IS 계정을 못 찾았으면 별도재무제표로 재시도
        if fs_div == "CFS":
            return get_annual_financials(api_key, corp_code, year, fs_div="OFS")
        return None

    debt_ratio = (liabilities / equity * 100) if (liabilities is not None and equity) else None
    roe = (net_income / equity * 100) if (net_income is not None and equity) else None

    return {
        "year": year,
        "revenue": revenue,
        "operating_profit": op_profit,
        "net_income": net_income,
        "total_equity": equity,
        "total_liabilities": liabilities,
        "debt_ratio": debt_ratio,
        "roe": roe,
        "fs_div": fs_div,
    }


def get_financial_summary(api_key: str, stock_code: str, corp_code_map: dict, years: list):
    corp_code = corp_code_map.get(stock_code)
    if not corp_code:
        raise DartApiError(f"종목코드 {stock_code}에 대한 DART corp_code를 찾을 수 없습니다.")

    rows = []
    for y in years:
        result = get_annual_financials(api_key, corp_code, y)
        if result:
            rows.append(result)
    return rows
