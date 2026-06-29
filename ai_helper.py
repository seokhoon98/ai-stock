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
                return "".join(p.get("text", "") for p in parts).strip()
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


def generate_financial_commentary(api_key: str, model: str, stock_name: str, financials: list) -> str:
    """핵심 재무 데이터를 바탕으로 한 줄 코멘터리 생성."""
    if not financials:
        return "분석에 사용할 재무 데이터가 부족합니다."

    rows_text = "\n".join(
        f"- {f['year']}년: 매출 {f.get('revenue')}, 영업이익 {f.get('operating_profit')}, "
        f"순이익 {f.get('net_income')}, 부채비율 {f.get('debt_ratio')}, ROE {f.get('roe')}"
        for f in financials
    )
    prompt = (
        f"다음은 {stock_name}의 최근 연간 재무 데이터(단위: 원, 비율은 %)이다.\n{rows_text}\n\n"
        "이 데이터를 바탕으로 매출/이익 추세, 재무 건전성(부채비율), 수익성(ROE) 관점에서 "
        "3~4문장으로 간결하게 한국어로 요약 코멘트를 작성해줘. 과장된 표현은 피하고 "
        "사실 기반으로 작성해줘."
    )
    commentary = _generate(api_key, model, prompt)
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
