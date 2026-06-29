"""
한국투자증권(KIS) Open API 클라이언트.

- OAuth 접근토큰 발급/캐시
- 주식잔고조회 (TTTC8434R / VTTC8434R)
- 주식현재가 시세조회 (FHKST01010100)

다른 사용자가 자신의 앱키/앱시크릿/계좌번호를 직접 입력해서 쓸 수 있도록,
이 클래스는 생성 시점에 자격증명을 인자로 받는다 (코드에 하드코딩하지 않음).
"""

import time
import requests

REAL_BASE_URL = "https://openapi.koreainvestment.com:9443"
VIRTUAL_BASE_URL = "https://openapivts.koreainvestment.com:29443"

RATE_LIMIT_ERROR_CODE = "EGW00201"  # 초당 거래건수 초과


class KISApiError(Exception):
    pass


class KISClient:
    def __init__(self, app_key: str, app_secret: str, account_no: str, is_virtual: bool = False):
        """
        account_no: "12345678-01" 또는 "1234567801" 형식의 계좌번호.
                    앞 8자리가 CANO, 뒤 2자리가 ACNT_PRDT_CD로 자동 분리된다.
        is_virtual: True면 모의투자 서버/거래ID를 사용한다.
        """
        self.app_key = app_key
        self.app_secret = app_secret
        self.is_virtual = is_virtual
        self.base_url = VIRTUAL_BASE_URL if is_virtual else REAL_BASE_URL

        digits = account_no.replace("-", "").strip()
        if len(digits) < 10:
            raise ValueError("계좌번호 형식이 올바르지 않습니다 (예: 12345678-01)")
        self.cano = digits[:8]
        self.acnt_prdt_cd = digits[8:10]

        self._access_token = None
        self._token_expires_at = 0

    # ------------------------------------------------------------------
    # 인증
    # ------------------------------------------------------------------
    def _ensure_token(self):
        if self._access_token and time.time() < self._token_expires_at - 60:
            return self._access_token

        url = f"{self.base_url}/oauth2/tokenP"
        body = {
            "grant_type": "client_credentials",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
        }
        resp = requests.post(url, json=body, timeout=10)
        try:
            data = resp.json()
        except ValueError:
            raise KISApiError(f"토큰 발급 응답 파싱 실패 (HTTP {resp.status_code}): {resp.text[:200]}")

        if resp.status_code != 200 or "access_token" not in data:
            msg = data.get("error_description") or data.get("msg1") or data
            raise KISApiError(
                f"토큰 발급 실패 (HTTP {resp.status_code}): {msg} "
                "— 앱키/시크릿이 올바른지, 토큰을 너무 짧은 간격으로 재발급하지 않았는지 확인하세요."
            )

        self._access_token = data["access_token"]
        # expires_in은 보통 86400(초). 약간 여유를 두고 만료시간을 기록.
        self._token_expires_at = time.time() + int(data.get("expires_in", 86400))
        return self._access_token

    def _headers(self, tr_id: str):
        token = self._ensure_token()
        return {
            "content-type": "application/json; charset=utf-8",
            "authorization": f"Bearer {token}",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
            "tr_id": tr_id,
            "custtype": "P",
        }

    def _get(self, path: str, tr_id: str, params: dict, retries: int = 3):
        url = f"{self.base_url}{path}"
        resp = requests.get(url, headers=self._headers(tr_id), params=params, timeout=10)
        try:
            data = resp.json()
        except ValueError:
            resp.raise_for_status()
            raise KISApiError(f"응답 파싱 실패: {resp.text[:200]}")

        if data.get("rt_cd") != "0":
            msg_cd = data.get("msg_cd", "")
            if retries > 0 and msg_cd == RATE_LIMIT_ERROR_CODE:
                time.sleep(0.7)
                return self._get(path, tr_id, params, retries=retries - 1)
            raise KISApiError(f"[{msg_cd}] {data.get('msg1', '알 수 없는 오류')}")
        return data

    # ------------------------------------------------------------------
    # 잔고조회
    # ------------------------------------------------------------------
    def get_balance(self):
        tr_id = "VTTC8434R" if self.is_virtual else "TTTC8434R"
        params = {
            "CANO": self.cano,
            "ACNT_PRDT_CD": self.acnt_prdt_cd,
            "AFHR_FLPR_YN": "N",
            "OFL_YN": "",
            "INQR_DVSN": "02",
            "UNPR_DVSN": "01",
            "FUND_STTL_ICLD_YN": "N",
            "FNCG_AMT_AUTO_RDPT_YN": "N",
            "PRCS_DVSN": "00",  # 00: 전일매매 포함 (당일 체결분도 반영)
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": "",
        }
        data = self._get("/uapi/domestic-stock/v1/trading/inquire-balance", tr_id, params)
        holdings = [h for h in data.get("output1", []) if int(h.get("hldg_qty", "0") or 0) > 0]
        summary = (data.get("output2") or [{}])[0]
        return holdings, summary

    # ------------------------------------------------------------------
    # 현재가 시세조회
    # ------------------------------------------------------------------
    def get_price(self, symbol: str):
        tr_id = "FHKST01010100"
        params = {"fid_cond_mrkt_div_code": "J", "fid_input_iscd": symbol}
        data = self._get("/uapi/domestic-stock/v1/quotations/inquire-price", tr_id, params)
        return data.get("output", {})
