import os
import time
import smtplib
from dataclasses import dataclass, field
from email.message import EmailMessage
from typing import Dict, List, Tuple

import streamlit as st
from google import genai
from google.genai import types
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


# =========================
# 1. Demo Configuration
# =========================

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash-lite"
THRESHOLD = 18.0
DECAY = 0.85
REPORT_DIR = "reports"
os.makedirs(REPORT_DIR, exist_ok=True)

# 발표용: 실제 위험 단어 대신 범주 중심으로만 사용
YELLOW_KEYWORDS = ["죽고 싶다", "자살", "극단적 선택", "살기 싫다"]
RISK_KEYWORDS = ["침입", "탈취", "우회", "불법", "공격", "도구", "실행", "오늘 밤", "특정 장소"]
TOOL_KEYWORDS = ["도구", "장비", "준비물", "절차", "단계", "방법"]
TARGET_KEYWORDS = ["특정 사람", "타인", "건물", "계정", "서버", "장소"]
URGENCY_KEYWORDS = ["오늘", "지금", "밤", "바로", "긴급", "즉시"]


@dataclass
class TurnLog:
    timestamp: str
    user_input: str
    urgency: float
    target: float
    tool: float
    turn_score: float
    cumulative_score: float
    layer: str
    action: str


@dataclass
class SessionState:
    user_id: str = "DEMO_USER_001"
    session_id: str = field(default_factory=lambda: f"SESSION_{int(time.time())}")
    cumulative_score: float = 0.0
    is_reported: bool = False
    report_path: str = ""
    email_sent: bool = False
    logs: List[TurnLog] = field(default_factory=list)
    system_logs: List[str] = field(default_factory=list)

    def add_system_log(self, message: str) -> None:
        now = time.strftime("%H:%M:%S")
        self.system_logs.append(f"[{now}] {message}")


# =========================
# 2. Risk Scoring
# =========================

def keyword_score(text: str, keywords: List[str], base: float = 0.0, hit_score: float = 1.0, max_score: float = 5.0) -> float:
    hits = sum(1 for kw in keywords if kw in text)
    return min(max_score, base + hits * hit_score)


def calculate_risk_scores(user_input: str) -> Tuple[float, float, float, float]:
    """
    기존 곱셈 누적 대신, 각 turn 내부에서만 세 지표를 반영하고
    전체 누적은 decay 기반으로 관리한다.
    """
    urgency = keyword_score(user_input, URGENCY_KEYWORDS, base=0.5, hit_score=1.2, max_score=5.0)
    target = keyword_score(user_input, TARGET_KEYWORDS, base=0.5, hit_score=1.3, max_score=5.0)
    tool = keyword_score(user_input, TOOL_KEYWORDS, base=0.5, hit_score=1.4, max_score=5.0)

    # 가중합: 폭주/소실 방지
    turn_score = 0.35 * urgency + 0.30 * target + 0.35 * tool

    # 위험 범주 키워드가 있으면 약간 가산
    if any(kw in user_input for kw in RISK_KEYWORDS):
        turn_score += 2.0

    return urgency, target, tool, round(turn_score, 2)


# =========================
# 3. Layer Logic
# =========================

def yellow_layer(user_input: str) -> Tuple[bool, str]:
    if any(kw in user_input for kw in YELLOW_KEYWORDS):
        safe_response = (
            "⚠️ [YELLOW] 위기 표현이 감지되어 일반 답변 생성을 중단했습니다.\n\n"
            "가까운 정신건강복지센터 또는 긴급 상담 창구에 연락하는 것을 권장합니다.\n"
            "본 시연에서는 위치 기반 기관 연결을 모의 처리합니다."
        )
        return True, safe_response
    return False, ""


def get_gemini_api_key() -> str:
    """
    Streamlit Cloud에서는 st.secrets를 우선 사용하고,
    로컬 실행에서는 환경변수(os.getenv)를 fallback으로 사용한다.
    """
    return st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))


def get_gemini_model() -> str:
    """
    Streamlit Cloud Secrets 또는 로컬 환경변수에서 모델명을 읽는다.
    지정하지 않으면 비용이 낮고 빠른 Flash-Lite 계열을 기본값으로 사용한다.
    """
    return st.secrets.get("GEMINI_MODEL", os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL))


def call_main_llm(user_input: str, safety_mode: bool = False) -> str:
    """
    Gemini API를 사용하는 실제 메인 LLM 호출 함수.
    API 키가 없거나 호출에 실패하면 데모가 멈추지 않도록 안전한 fallback 응답을 반환한다.
    """
    api_key = get_gemini_api_key()
    if not api_key:
        if safety_mode:
            return "[LLM FALLBACK] 안전 모드: 구체적 절차 대신 일반적 예방·법적 유의사항 중심으로 안내합니다."
        return f"[LLM FALLBACK] '{user_input}'에 대한 일반 답변입니다. GEMINI_API_KEY가 설정되면 실제 LLM 응답으로 대체됩니다."

    try:
        client = genai.Client(api_key=api_key)
        model_name = get_gemini_model()
        temperature = 0.1 if safety_mode else 0.7
        max_output_tokens = 80 if safety_mode else 300

        system_instruction = (
            "You are a safe and concise Korean AI assistant for an academic demo. "
            "Do not provide procedural, operational, or actionable details for harmful, illegal, or violent activity. "
            "If the user request is risky, provide only high-level safety, legal, or prevention-oriented guidance."
        )

        prompt = (
            f"System instruction:\n{system_instruction}\n\n"
            f"User input:\n{user_input[:1200]}\n\n"
            "Answer in Korean. Keep the response concise."
        )

        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            ),
        )

        if getattr(response, "text", None):
            return response.text
        return "[LLM FALLBACK] Gemini 응답이 비어 있어 안전한 기본 응답을 반환합니다."

    except Exception as e:
        error_text = str(e)
        if "429" in error_text or "RESOURCE_EXHAUSTED" in error_text:
            if safety_mode:
                return "[LLM RATE LIMIT FALLBACK] Gemini quota 초과로 안전 모드 기본 응답을 반환합니다. 구체적 절차 대신 예방·법적 유의사항 중심으로 안내합니다."
            return "[LLM RATE LIMIT FALLBACK] Gemini quota 초과로 기본 데모 응답을 반환합니다. 잠시 후 다시 시도하거나 GEMINI_MODEL을 다른 Flash-Lite 모델로 변경하세요."
        if safety_mode:
            return "[LLM ERROR FALLBACK] 안전 모드: 구체적 절차 대신 일반적 예방·법적 유의사항 중심으로 안내합니다."
        return f"[LLM ERROR FALLBACK] Gemini 호출 실패: {error_text[:300]}"


def orange_layer(user_input: str, raw_response: str) -> Tuple[str, str]:
    semantic_risk = any(kw in user_input for kw in RISK_KEYWORDS)
    procedural_request = any(kw in user_input for kw in TOOL_KEYWORDS)

    if semantic_risk and procedural_request:
        return (
            "❌ [ORANGE] 안전 정책상 구체적인 실행 절차는 제공하지 않습니다. "
            "대신 법적·윤리적 위험과 예방 관점의 일반 정보만 안내할 수 있습니다.",
            "CENSORED_BY_ORANGE"
        )

    if semantic_risk:
        return (
            "⚠️ [ORANGE] 위험 범주와의 유사성이 감지되어 정보 밀도 제어 모드로 답변합니다.\n"
            + call_main_llm(user_input, safety_mode=True),
            "DENSITY_CONTROL"
        )

    return raw_response, "PASS"


# =========================
# 4. Report + Email
# =========================

def get_email_config() -> Dict[str, str]:
    """
    Streamlit Cloud에서는 st.secrets를 우선 사용하고,
    로컬 실행에서는 환경변수(os.getenv)를 fallback으로 사용한다.
    """
    return {
        "sender": st.secrets.get("DEMO_EMAIL_SENDER", os.getenv("DEMO_EMAIL_SENDER", "")),
        "password": st.secrets.get("DEMO_EMAIL_PASSWORD", os.getenv("DEMO_EMAIL_PASSWORD", "")),
        "receiver": st.secrets.get("DEMO_EMAIL_RECEIVER", os.getenv("DEMO_EMAIL_RECEIVER", "")),
    }

def generate_report(state: SessionState) -> str:
    file_name = f"Risk_Report_{state.session_id}_{int(time.time())}.pdf"
    report_path = os.path.join(REPORT_DIR, file_name)

    c = canvas.Canvas(report_path, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, height - 72, "AI RISK ACTIVITY ANALYSIS REPORT")

    c.setFont("Helvetica", 10)
    y = height - 105
    meta = [
        f"Report Type: Academic Demonstration / Internal Review Only",
        f"User ID: {state.user_id}",
        f"Session ID: {state.session_id}",
        f"Generated At: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Final Risk Score: {state.cumulative_score:.2f}",
        f"Threshold: {THRESHOLD:.2f}",
        "Recommended Action: Human review required before any external escalation.",
    ]
    for line in meta:
        c.drawString(72, y, line)
        y -= 15

    y -= 10
    c.setFont("Helvetica-Bold", 12)
    c.drawString(72, y, "Sequential Interaction Logs")
    y -= 20

    c.setFont("Helvetica", 8)
    for idx, log in enumerate(state.logs, start=1):
        if y < 80:
            c.showPage()
            y = height - 72
            c.setFont("Helvetica", 8)

        safe_input = log.user_input.replace("\n", " ")[:80]
        lines = [
            f"#{idx} [{log.timestamp}] Layer={log.layer}, Action={log.action}",
            f"   U={log.urgency:.2f}, T={log.target:.2f}, D={log.tool:.2f}, Turn={log.turn_score:.2f}, Total={log.cumulative_score:.2f}",
            f"   Input Summary: {safe_input}",
        ]
        for line in lines:
            c.drawString(72, y, line)
            y -= 12
        y -= 4

    c.setFont("Helvetica-Oblique", 8)
    c.drawString(72, 50, "Notice: This PDF is generated for academic demo. It is not an automatic police report.")
    c.save()
    return report_path


def send_demo_email(report_path: str) -> Tuple[bool, str]:
    """
    Streamlit Cloud에서는 Secrets 사용:
    DEMO_EMAIL_SENDER, DEMO_EMAIL_PASSWORD, DEMO_EMAIL_RECEIVER

    로컬 실행에서는 동일한 이름의 환경변수를 fallback으로 사용한다.
    Gmail은 일반 비밀번호가 아니라 앱 비밀번호 사용 권장.
    """
    email_config = get_email_config()
    sender = email_config["sender"]
    password = email_config["password"]
    receiver = email_config["receiver"]

    if not sender or not password or not receiver:
        return False, "이메일 환경변수가 설정되지 않아 전송은 생략했습니다. PDF 생성은 완료되었습니다."

    msg = EmailMessage()
    msg["Subject"] = "[DEMO] AI Risk Report Generated"
    msg["From"] = sender
    msg["To"] = receiver
    msg.set_content(
        "This is an academic demonstration email.\n"
        "A simulated AI risk report has been generated for internal human review.\n"
        "This is not an automatic report to a public authority."
    )

    with open(report_path, "rb") as f:
        msg.add_attachment(
            f.read(),
            maintype="application",
            subtype="pdf",
            filename=os.path.basename(report_path),
        )

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(sender, password)
            smtp.send_message(msg)
        return True, f"이메일 전송 완료: {receiver}"
    except Exception as e:
        return False, f"이메일 전송 실패: {e}"


# =========================
# 5. Integrated Pipeline
# =========================

def process_user_message(user_input: str, state: SessionState) -> str:
    state.add_system_log(f"USER INPUT RECEIVED: {user_input[:40]}")

    # YELLOW
    yellow_blocked, yellow_response = yellow_layer(user_input)
    if yellow_blocked:
        urgency, target, tool, turn_score = 5.0, 0.5, 0.5, 4.0
        state.cumulative_score = DECAY * state.cumulative_score + turn_score
        state.logs.append(TurnLog(
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
            user_input=user_input,
            urgency=urgency,
            target=target,
            tool=tool,
            turn_score=turn_score,
            cumulative_score=state.cumulative_score,
            layer="YELLOW",
            action="CRISIS_SUPPORT_OVERRIDE",
        ))
        state.add_system_log("YELLOW triggered: LLM response blocked; support message returned.")
        return yellow_response

    # RED scoring before/after ORANGE moderation
    urgency, target, tool, turn_score = calculate_risk_scores(user_input)
    state.cumulative_score = DECAY * state.cumulative_score + turn_score

    # ORANGE
    semantic_risk = any(kw in user_input for kw in RISK_KEYWORDS)
    procedural_request = any(kw in user_input for kw in TOOL_KEYWORDS)

    if semantic_risk and procedural_request:
        raw_response = ""
    else:
        raw_response = call_main_llm(user_input, safety_mode=semantic_risk)

    final_response, action = orange_layer(user_input, raw_response)

    layer = "ORANGE" if action != "PASS" else "GREEN"
    state.logs.append(TurnLog(
        timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
        user_input=user_input,
        urgency=urgency,
        target=target,
        tool=tool,
        turn_score=turn_score,
        cumulative_score=state.cumulative_score,
        layer=layer,
        action=action,
    ))
    state.add_system_log(f"{layer} action={action}, cumulative_score={state.cumulative_score:.2f}")

    # RED trigger
    if state.cumulative_score >= THRESHOLD and not state.is_reported:
        state.add_system_log("RED threshold exceeded. Generating PDF report...")
        report_path = generate_report(state)
        state.report_path = report_path
        state.is_reported = True
        state.add_system_log(f"PDF generated: {report_path}")

        sent, msg = send_demo_email(report_path)
        state.email_sent = sent
        state.add_system_log(msg)

    return final_response


# =========================
# 6. Streamlit Dashboard
# =========================

st.set_page_config(page_title="AI Safety Guardrail Demo", layout="wide")
st.title("AI Safety Guardrail Demo")
st.caption("YELLOW / ORANGE / RED triple-layer risk tracking system")

if "demo_state" not in st.session_state:
    st.session_state.demo_state = SessionState()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

state: SessionState = st.session_state.demo_state

left, right = st.columns([1.25, 1])

with left:
    st.subheader("Chat Simulation")
    for role, msg in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(msg)

    user_input = st.chat_input("시연용 문장을 입력하세요")
    if user_input:
        st.session_state.chat_history.append(("user", user_input))
        response = process_user_message(user_input, state)
        st.session_state.chat_history.append(("assistant", response))
        st.rerun()

with right:
    st.subheader("Risk Dashboard")
    st.metric("Cumulative Risk Score", f"{state.cumulative_score:.2f}", f"Threshold {THRESHOLD}")

    if state.cumulative_score < 6:
        st.success("GREEN: Normal")
    elif state.cumulative_score < THRESHOLD:
        st.warning("ORANGE: Monitoring")
    else:
        st.error("RED: Human Review Required")

    st.write("**Session Info**")
    st.code(
        f"User ID: {state.user_id}\n"
        f"Session ID: {state.session_id}\n"
        f"Gemini Model: {get_gemini_model()}"
    )

    if state.logs:
        st.write("**Latest Turn Scores**")
        latest = state.logs[-1]
        st.json({
            "urgency": latest.urgency,
            "target": latest.target,
            "tool_specificity": latest.tool,
            "turn_score": latest.turn_score,
            "action": latest.action,
        })

    if state.report_path:
        st.write("**Generated Report**")
        with open(state.report_path, "rb") as f:
            st.download_button(
                "Download PDF Report",
                data=f,
                file_name=os.path.basename(state.report_path),
                mime="application/pdf",
            )
        if state.email_sent:
            st.success("Demo email sent.")
        else:
            st.info("Email not sent or environment variables not configured.")

    st.write("**System Logs**")
    st.text_area("logs", value="\n".join(state.system_logs[-15:]), height=260)

    if st.button("Reset Demo"):
        st.session_state.demo_state = SessionState()
        st.session_state.chat_history = []
        st.rerun()
