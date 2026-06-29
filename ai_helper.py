"""
Gemini API(REST) 래퍼.

SDK 의존성 없이 raw REST 호출만 사용한다 (모델명/버전이 바뀌어도
requirements.txt를 건드릴 필요가 없도록). 모델명은 사이드바에서
사용자가 직접 지정할 수 있게 한다 — 학습 시점 이후 모델명이
바뀌었을 가능성이 있기 때문.

이 모듈이 생성하는 코멘트/제안은 전부 "AI가 생성한 참고용 의견이며
투자 자문이 아님"을 명시한다.
"""

import json
import re
import time

import requests

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
DISCLAIMER = "⚠️ 위 내용은 AI가 생성한 참고용 의견이며, 투자 자문이 아닙니다. 투자 결정과 책임은 본인에게 있습니다."

RETRYABLE_STATUS_CODES = {429, 503}
MAX_RETRIES = 2

# 세션 내 누적 사용량 (app.py에서 st.session_state로 관리)
_session_usage: dict = {
    "calls": 0,
    "prompt_tokens": 0,
    "output_tokens": 0,
    "total_tokens": 0,
}


def get_usage() -> dict:
    return dict(_session_usage)


def reset_usage():
    _session_usage.update(calls=0, prompt_tokens=0, output_tokens=0, total_tokens=0)


class AIHelperError(Exception):
    pass


def _parse_retry_delay(err: dict, fallback: float) -> float:
    """429 응답의 RetryInfo.retryDelay(예: "46s")를 읽어 대기 시간을 정한다."""
    try:
        for detail in err.get("error", {}).get("details", []):
            if detail.get("@type", "").endswith("RetryInfo"):
                delay = detail.get("retryDelay", "")
                if delay.endswith("s"):
                    return float(delay[:-1])
    except (AttributeError, ValueError):
        pass
    return fallback


def _try_single_key(api_key: str, model: str, payload: dict, timeout: int) -> str:
    """단일 키로 최대 MAX_RETRIES+1회 시도. 429/503이면 AIHelperError(rate_limit=True) 발생."""
    url = f"{GEMINI_BASE}/{model}:generateContent"
    last_err = None

    for attempt in range(MAX_RETRIES + 1):
        resp = requests.post(url, params={"key": api_key}, json=payload, timeout=timeout)

        if resp.status_code == 200:
            data = resp.json()
            try:
                candidates = data["candidates"]
                parts = candidates[0]["content"]["parts"]
                text = "".join(p.get("text", "") for p in parts).strip()
                # 사용량 누적
                usage = data.get("usageMetadata", {})
                _session_usage["calls"] += 1
                _session_usage["prompt_tokens"]  += usage.get("promptTokenCount", 0)
                _session_usage["output_tokens"]  += usage.get("candidatesTokenCount", 0)
                _session_usage["total_tokens"]   += usage.get("totalTokenCount", 0)
                return text
            except (KeyError, IndexError):
                raise AIHelperError(f"Gemini 응답 파싱 실패: {data}")

        try:
            err = resp.json()
        except ValueError:
            err = resp.text

        last_err = (resp.status_code, err)

        if resp.status_code in RETRYABLE_STATUS_CODES and attempt < MAX_RETRIES:
            wait = _parse_retry_delay(err if isinstance(err, dict) else {}, fallback=2 ** attempt * 3)
            time.sleep(min(wait, 15))
            continue
        break

    status_code, err = last_err
    exc = AIHelperError(f"Gemini API 오류({status_code}): {err}")
    exc.rate_limited = (status_code == 429)
    raise exc


def _generate(api_key: str | list, model: str, prompt: str, timeout: int = 30) -> str:
    """
    api_key가 문자열이면 단일 키, 리스트면 429 발생 시 다음 키로 순환한다.
    모든 키가 소진되면 마지막 오류를 그대로 올린다.
    """
    keys = [api_key] if isinstance(api_key, str) else list(api_key)
    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    last_exc = None
    for i, key in enumerate(keys):
        try:
            return _try_single_key(key, model, payload, timeout)
        except AIHelperError as e:
            last_exc = e
            if getattr(e, "rate_limited", False) and i < len(keys) - 1:
                # 429 → 다음 키로 즉시 전환 (짧은 대기 없음)
                continue
            raise

    raise last_exc


def generate_financial_commentary(
    api_key: str, model: str, stock_name: str, financials: list, valuation: dict = None
) -> str:
    """애널리스트 리포트 수준의 재무 분석 코멘터리 생성."""
    if not financials:
        return "분석에 사용할 재무 데이터가 부족합니다."

    def fmt(v, unit="억원", div=1e8):
        if v is None:
            return "N/A"
        try:
            return f"{v/div:,.0f}{unit}"
        except Exception:
            return str(v)

    def pct(v):
        return f"{v:.1f}%" if v is not None else "N/A"

    rows_text = "\n".join(
        f"- {f['year']}년: 매출 {fmt(f.get('revenue'))}, 영업이익 {fmt(f.get('operating_profit'))}, "
        f"순이익 {fmt(f.get('net_income'))}, 영업이익률 {pct(f.get('op_margin'))}, "
        f"부채비율 {pct(f.get('debt_ratio'))}, ROE {pct(f.get('roe'))}, ROIC {pct(f.get('roic'))}"
        for f in financials
    )

    val_text = ""
    if valuation:
        val_text = (
            f"\n[현재 밸류에이션]\n"
            f"- PER: {pct(valuation.get('per'))} (선행 PER: {pct(valuation.get('forward_per'))})\n"
            f"- PBR: {valuation.get('pbr'):.2f}배\n" if valuation.get('pbr') else ""
            f"- EPS: {fmt(valuation.get('eps'), '원', 1)}\n"
            f"- 배당수익률: {pct(valuation.get('dividend_yield'))}"
        ) if valuation else ""

    prompt = f"""당신은 국내 증권사 리서치센터 소속 주식 애널리스트입니다.
아래 {stock_name}의 재무 데이터를 바탕으로 기관투자자 수준의 심층 분석 리포트를 한국어로 작성하세요.

[연간 재무 데이터 (단위: 억원, 비율: %)]
{rows_text}
{val_text}

다음 구조로 작성하되, 각 항목은 구체적인 수치를 인용하며 분석하세요:

## 1. 매출 및 이익 성장성
- 매출 CAGR 및 성장 트렌드 분석
- 영업이익·순이익 증감 원인 추론 (마진 압박 or 확장 여부)

## 2. 수익성 분석
- 영업이익률 추이 및 업종 내 포지셔닝
- ROE·ROIC를 통한 자본 효율성 평가
- ROIC vs 자본비용(WACC 추정 8~10%) 비교

## 3. 재무 건전성
- 부채비율 추이 및 재무 레버리지 평가
- 이익 대비 부채 수준의 적정성

## 4. 밸류에이션
- PER·PBR 기준 현재 주가 수준 평가 (저평가/적정/고평가)
- 성장성과 수익성 대비 밸류에이션 매력도

## 5. 투자 의견 요약
- 핵심 강점 2~3가지
- 주요 리스크 2~3가지
- 종합 의견 (매수 검토 / 관망 / 주의) — 단, 확정적 매수·매도 권유는 하지 말 것

수치는 반드시 데이터에 근거하고, 추론 시 "추정된다", "판단된다" 등의 표현을 사용하세요.
분량은 600~900자 내외로 작성하세요."""

    commentary = _generate(api_key, model, prompt, timeout=60)
    return f"{commentary}\n\n{DISCLAIMER}"


def generate_rebalancing_table(
    api_key: str,
    model: str,
    holdings: list,
    watchlist: list,
    account_summary: dict,
) -> list:
    """
    보유 종목 + 관심종목을 바탕으로 리밸런싱 제안을 구조화된 JSON 목록으로 반환한다.
    반환: [{"symbol": "005380", "name": "현대차", "action": "매도", "qty": 2, "reason": "..."}]
    action 값은 반드시 "매수 제안" 또는 "매도 제안" 중 하나.
    """
    holdings_text = "\n".join(
        f"- {h.get('name')}({h.get('symbol')}): {h.get('qty')}주, "
        f"평가금액 {h.get('eval_amount')}원, 평가수익률 {h.get('profit_rate')}%"
        for h in holdings
    ) or "보유 종목 없음"

    watchlist_text = "\n".join(
        f"- {w.get('name')}({w.get('symbol')}): 현재가 {w.get('price')}원"
        for w in watchlist
    ) or "관심종목 없음"

    prompt = (
        "당신은 포트폴리오 리밸런싱을 보조하는 분석 도구입니다. "
        "아래 데이터를 참고해서 포트폴리오 다각화와 리스크 관리 관점의 매매 제안을 만들어주세요.\n\n"
        f"[보유 종목]\n{holdings_text}\n\n"
        f"[관심 종목 워치리스트]\n{watchlist_text}\n\n"
        f"[계좌 요약]\n총평가금액: {account_summary.get('total_eval_amount', 'N/A')}원, "
        f"총평가수익률: {account_summary.get('total_profit_rate', 'N/A')}%\n\n"
        "3~5개의 구체적인 종목별 매수/매도 제안을 만들어주세요. "
        "매수 제안은 워치리스트 종목 중에서, 매도/비중축소는 보유 종목 중에서 골라주세요. "
        "수량(qty)은 해당 액션을 취할 주수(양의 정수)로 써주세요. "
        "확정적 권유가 아니라 참고 의견임을 reason에 반영해주세요.\n\n"
        "다른 설명 없이 아래 JSON 배열 형식으로만 답변해주세요:\n"
        '[{"symbol": "6자리종목코드", "name": "종목명", "action": "매수 제안" 또는 "매도 제안", '
        '"qty": 수량(정수), "reason": "사유 한 문장"}, ...]'
    )
    raw = _generate(api_key, model, prompt)
    parsed = _extract_json(raw)
    if not isinstance(parsed, list):
        raise AIHelperError(f"AI 응답이 목록 형식이 아닙니다: {raw[:300]}")
    return parsed


def _extract_json(text: str):
    """Gemini가 ```json ... ``` 코드블록으로 감싸서 응답하는 경우가 많아 벗겨낸다."""
    cleaned = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", cleaned, re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1)
    else:
        bracket_match = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if bracket_match:
            cleaned = bracket_match.group(0)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise AIHelperError(f"AI 응답을 종목 목록(JSON)으로 해석하지 못했습니다: {e}\n원본: {text[:300]}")


# 투자 성향 설문에서 쓰는 선택지 (idea-generation 스킬의 Step 1 기준과 동일하게 구성)
DIRECTION_OPTIONS = ["롱(매수) 아이디어만", "숏(매도) 아이디어만", "롱/숏 둘 다"]
MARKET_CAP_OPTIONS = ["대형주", "중형주", "소형주", "무관"]
STYLE_OPTIONS = ["성장주(Growth)", "가치주(Value)", "퀄리티(Quality)", "이벤트 드리븐/특수상황"]


def generate_watchlist_suggestions(
    api_key: str,
    model: str,
    direction: str,
    market_cap: str,
    style: str,
    themes: str,
    exclude_names: list = None,
    count: int = 5,
) -> list:
    """
    투자 성향 설문 답변을 바탕으로 한국(KOSPI/KOSDAQ) 종목 후보를 추천받는다.
    반환: [{"symbol": "005930", "name": "삼성전자", "theme": "...", "reason": "..."}, ...]

    주의: Gemini는 실시간 시세/최신 상장 정보를 보장하지 않으므로, 반환된 종목코드는
    호출 측(app.py)에서 KIS 현재가 조회로 검증 후 사용해야 한다.
    """
    exclude_text = ", ".join(exclude_names) if exclude_names else "없음"
    prompt = (
        "당신은 한국 주식시장(KOSPI/KOSDAQ) 종목을 추천하는 분석 보조 도구입니다. "
        "아래 투자 성향 설문 답변에 맞는 신규 관심종목(워치리스트) 후보를 추천해주세요.\n\n"
        f"- 방향: {direction}\n"
        f"- 시가총액 선호: {market_cap}\n"
        f"- 투자 스타일: {style}\n"
        f"- 관심 섹터/테마: {themes or '특정 선호 없음, 시장 주도 테마 중 추천'}\n"
        f"- 이미 보유 중이거나 기존 워치리스트에 있어 제외해야 할 종목: {exclude_text}\n\n"
        f"정확히 {count}개의 종목을 추천하고, 위에서 제외 대상으로 언급된 종목과 섹터가 "
        "겹치지 않게 다양화해주세요. 반드시 실제 존재하는 한국 상장사와 정확한 6자리 "
        "종목코드를 사용해주세요(불확실하면 추측하지 말고 더 확실히 아는 종목으로 대체).\n\n"
        "다른 설명 없이 아래 JSON 배열 형식으로만 답변해주세요:\n"
        '[{"symbol": "6자리 종목코드", "name": "종목명", "theme": "테마/섹터", '
        '"reason": "추천 이유 한 문장"}, ...]'
    )
    raw = _generate(api_key, model, prompt)
    parsed = _extract_json(raw)
    if not isinstance(parsed, list):
        raise AIHelperError(f"AI 응답이 목록 형식이 아닙니다: {raw[:300]}")
    return parsed
