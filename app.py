"""AI 관상가 - 얼굴 사진을 보고 관상을 봐주는 Streamlit 앱."""

import base64
import hashlib
import io
import os
from pathlib import Path

import openai
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image, ImageOps, UnidentifiedImageError

DEFAULT_MODEL = "gpt-5.4-mini"
ALLOWED_TYPES = ["jpg", "jpeg", "png", "webp"]
MAX_FILE_BYTES = 10 * 1024 * 1024  # 10MB
MAX_IMAGE_SIDE = 1024
MAX_COMPLETION_TOKENS = 2500  # 추론 모델은 추론 토큰도 여기에 포함된다
API_TIMEOUT_SECONDS = 60  # SDK 기본값(600초 x 재시도 2회)은 너무 길다
API_MAX_RETRIES = 1

DISCLAIMER = (
    "본 결과는 재미를 위한 것이며 과학적 근거가 없습니다. "
    "업로드한 사진은 저장되지 않습니다."
)

SYSTEM_PROMPT = """당신은 40년 경력의 한국 전통 관상가입니다.
사용자가 재미로 관상을 보고 싶어서 본인(또는 동의를 받은 지인)의 얼굴 사진을 직접 올렸습니다.
사진에서 보이는 이마, 눈썹, 눈, 코, 입, 귀, 턱의 생김새를 묘사하고,
그 생김새를 전통 관상학에 따라 풀이한 오락용 해석을 한국어로 들려줍니다.
이것은 신원 확인이나 외모 평가가 아니라 운세 콘텐츠이므로, 얼굴이 보이면 반드시 풀이해주세요.

[출력 형식] 반드시 아래 Markdown 구조를 그대로 따르세요.

## ✨ 첫인상 요약
(한 줄 요약)

## 👁️ 부위별 관상
### 이마 (초년운·지혜)
### 눈썹 (형제운·성격)
### 눈 (마음·대인관계)
### 코 (재물운·중년운)
### 입·입술 (언변·식복)
### 귀 (복·수명)
### 턱·얼굴형 (말년운·의지)
(각 부위마다 2~3문장. 사진에서 잘 보이지 않는 부위는 "사진상 잘 보이지 않아 일반적인 해석을 드립니다"라고 밝히고 짧게 해석)

## 🔮 종합 운세
| 운세 | 점수 | 한마디 |
|---|---|---|
| 재물운 | ★★★★☆ | ... |
| 연애운 | ... | ... |
| 직업운 | ... | ... |
| 건강운 | ... | ... |
(점수는 ★과 ☆로 5칸)

## 💼 어울리는 직업과 조언
- 어울리는 직업: (2~3개)
- 관상가의 한마디: (따뜻한 조언 한 문장)

[규칙]
- 긍정적이고 유쾌한 톤을 유지하고, 외모를 비하하거나 모욕하는 표현은 쓰지 마세요.
- 인종, 민족, 종교, 성적 지향, 건강 상태, 장애 등 민감한 속성은 추론하거나 언급하지 마세요.
  건강운은 관상학적 기운으로만 이야기하고 질병을 진단하지 마세요.
- 사진 속 인물이 누구인지 식별하거나 실존 인물과 연결하지 마세요.
  관상 풀이에는 신원 확인이 필요 없으므로, 신원을 모른다는 이유로 풀이를 거절하지 마세요.
- 이 해석은 전통 관상학에 기반한 재미용 콘텐츠입니다.
- 사진에 사람 얼굴이 전혀 없을 때(풍경, 사물, 동물 등)에만, 위 형식 대신 정확히
  "얼굴이 잘 보이는 사진을 올려주세요." 한 문장만 답하세요.
  사람 얼굴이 하나라도 보이면 이 문장을 쓰지 말고 반드시 위 형식으로 풀이하세요.
"""

USER_PROMPT = (
    "사진 속 인물이 누구인지는 알려주실 필요가 없습니다. "
    "이마, 눈썹, 눈, 코, 입, 귀, 턱의 생김새만 보고 "
    "정해진 형식에 따라 재미로 전통 관상 풀이를 해주세요."
)

NO_FACE_REPLY = "얼굴이 잘 보이는 사진을 올려주세요."
REFUSAL_MESSAGE = "AI가 이 사진의 관상 풀이를 거절했습니다. 잠시 후 다시 시도하거나 다른 사진을 올려주세요."
TRUNCATED_MESSAGE = "AI 답변이 너무 길어 완성되지 못했습니다. 「관상 보기」를 다시 눌러주세요."
TRUNCATED_NOTE = "\n\n---\n⚠️ *답변이 길어 중간에 잘렸습니다. 전체 풀이를 보려면 「관상 보기」를 다시 눌러주세요.*"
WARNING_RESULTS = (NO_FACE_REPLY, REFUSAL_MESSAGE, TRUNCATED_MESSAGE)


# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------
def get_config(name: str, default: str | None = None) -> str | None:
    """st.secrets(Streamlit Cloud) → .env(로컬) 순서로 설정값을 읽는다."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        # 로컬에 secrets.toml이 없으면 st.secrets 접근 시 예외가 발생한다.
        pass
    load_dotenv(Path(__file__).with_name(".env"))
    return os.getenv(name, default)


def get_api_key() -> str:
    api_key = get_config("OPENAI_API_KEY")
    if not api_key:
        st.error(
            "OpenAI API 키가 설정되지 않았습니다. "
            "로컬에서는 `.env` 파일에, Streamlit Cloud에서는 Secrets에 "
            "`OPENAI_API_KEY`를 등록해주세요."
        )
        st.stop()
    return api_key


def get_model() -> str:
    return get_config("OPENAI_MODEL") or DEFAULT_MODEL


# ---------------------------------------------------------------------------
# 이미지 처리 (메모리에서만 처리, 디스크 저장 없음)
# ---------------------------------------------------------------------------
def load_image(data: bytes) -> Image.Image | None:
    """바이트를 RGB PIL 이미지로 연다. 실패하면 에러를 표시하고 None."""
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except Image.DecompressionBombError:
        # 용량은 작아도 픽셀 수가 비정상적으로 큰 이미지 (메모리 폭탄 방지)
        st.error("이미지 해상도가 너무 큽니다. 일반적인 크기의 사진을 올려주세요.")
        return None
    except (UnidentifiedImageError, OSError, ValueError):
        st.error("이미지를 열 수 없습니다. 손상되지 않은 JPG/PNG/WEBP 파일인지 확인해주세요.")
        return None

    image = ImageOps.exif_transpose(image)  # 휴대폰 사진 회전 보정
    if image.mode != "RGB":
        if image.mode in ("RGBA", "LA", "P"):
            image = image.convert("RGBA")
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.getchannel("A"))
            image = background
        else:
            image = image.convert("RGB")
    return image


def resize_image(image: Image.Image) -> Image.Image:
    """긴 변 기준 MAX_IMAGE_SIDE 이하로 축소한다."""
    resized = image.copy()
    resized.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
    return resized


def to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


# ---------------------------------------------------------------------------
# OpenAI 분석
# ---------------------------------------------------------------------------
def analyze_face(image: Image.Image, api_key: str, model: str) -> str:
    """OpenAI Vision 모델로 관상을 분석해 Markdown 문자열을 반환한다."""
    client = OpenAI(api_key=api_key, timeout=API_TIMEOUT_SECONDS, max_retries=API_MAX_RETRIES)
    data_url = to_data_url(resize_image(image))
    response = client.chat.completions.create(
        model=model,
        max_completion_tokens=MAX_COMPLETION_TOKENS,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_PROMPT},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
    )
    choice = response.choices[0]
    message = choice.message
    content = (message.content or "").strip()
    truncated = choice.finish_reason == "length"

    # 모델이 거절하면 refusal 필드가 채워지거나, 형식 없는 짧은 거절문이 온다.
    if getattr(message, "refusal", None):
        return REFUSAL_MESSAGE
    if not content:
        # 추론 모델은 토큰을 추론에 다 써서 본문이 비어 있을 수 있다.
        return TRUNCATED_MESSAGE if truncated else REFUSAL_MESSAGE
    if "## " not in content:
        # 마침표·공백·이모지가 조금 달라도 "얼굴 없음" 안내로 인식한다.
        if NO_FACE_REPLY.rstrip(".") in content:
            return NO_FACE_REPLY
        return TRUNCATED_MESSAGE if truncated else REFUSAL_MESSAGE
    if truncated:
        return content + TRUNCATED_NOTE
    return content


def run_analysis(image: Image.Image, api_key: str, model: str) -> str | None:
    """analyze_face를 호출하고, 오류는 사용자 안내 메시지로 바꾼다."""
    try:
        return analyze_face(image, api_key, model)
    except openai.AuthenticationError:
        st.error("API 키가 올바르지 않습니다. `OPENAI_API_KEY` 값을 확인해주세요.")
    except openai.PermissionDeniedError:
        st.error("이 API 키로는 해당 모델을 사용할 권한이 없습니다.")
    except openai.NotFoundError:
        st.error(f"모델 `{model}`을(를) 찾을 수 없습니다. `OPENAI_MODEL` 값을 확인해주세요.")
    except openai.RateLimitError:
        st.error("요청이 너무 많거나 크레딧이 부족합니다. 잠시 후 다시 시도하거나 사용량을 확인해주세요.")
    except openai.APITimeoutError:
        st.error("AI 응답 시간이 초과되었습니다. 잠시 후 다시 시도해주세요.")
    except openai.APIConnectionError:
        st.error("OpenAI 서버에 연결할 수 없습니다. 인터넷 연결을 확인해주세요.")
    except openai.BadRequestError:
        st.error("요청을 처리할 수 없습니다. 다른 사진으로 다시 시도해주세요.")
    except openai.APIStatusError as e:
        st.error(f"OpenAI 서버 오류가 발생했습니다. (상태 코드 {e.status_code}) 잠시 후 다시 시도해주세요.")
    except Exception:
        st.error("알 수 없는 오류가 발생했습니다. 잠시 후 다시 시도해주세요.")
    return None


# ---------------------------------------------------------------------------
# 화면
# ---------------------------------------------------------------------------
def set_source(source: str) -> None:
    st.session_state.source = source


def get_input_bytes() -> bytes | None:
    """업로드/카메라 탭 중 마지막으로 입력된 이미지의 바이트를 반환한다."""
    upload_tab, camera_tab = st.tabs(["📁 사진 업로드", "📷 카메라 촬영"])
    with upload_tab:
        uploaded = st.file_uploader(
            "📂 사진을 아래 칸에 끌어다 놓거나, 칸을 눌러 폴더에서 선택하세요 (JPG·PNG·WEBP)",
            type=ALLOWED_TYPES,
            key="upload",
            on_change=set_source,
            args=("upload",),
        )
    with camera_tab:
        captured = st.camera_input(
            "카메라로 촬영하세요",
            key="camera",
            on_change=set_source,
            args=("camera",),
        )

    source = st.session_state.get("source")
    chosen = captured if source == "camera" else uploaded
    if chosen is None:  # 선택된 쪽이 비었으면 다른 쪽이라도 사용
        chosen = uploaded or captured
    if chosen is None:
        return None

    if chosen.size > MAX_FILE_BYTES:
        st.warning("파일이 너무 큽니다. 10MB 이하의 사진을 올려주세요.")
        return None
    return chosen.getvalue()


def render_result(result: str) -> None:
    if result in WARNING_RESULTS:
        st.warning(result)
        return
    st.markdown(result)
    # 결과가 길어 페이지 하단 면책 문구가 안 보일 수 있으므로 결과 끝에도 표시한다.
    st.caption(f"⚠️ {DISCLAIMER}")


# 테마(.streamlit/config.toml)로 표현이 안 되는 부분만 최소한으로 보정한다.
# 색은 Streamlit 테마 변수를 쓰므로 라이트/다크 모드 모두에서 읽힌다.
CUSTOM_CSS = """
<style>
.st-key-hero h1 { letter-spacing: 0.02em; }
.st-key-hero .seal {
    display: inline-block; padding: 0.15rem 0.55rem; margin-bottom: 0.4rem;
    border: 2px solid var(--primary-color, #B23A2E); border-radius: 4px;
    color: var(--primary-color, #B23A2E); font-weight: 700; font-size: 0.8rem;
    letter-spacing: 0.3em;
}
.st-key-result_card h2 { font-size: 1.3rem; margin-top: 1.2rem; }
.st-key-result_card h2:first-child { margin-top: 0; }
.st-key-result_card h3 { font-size: 1.05rem; margin-top: 0.8rem; }
.st-key-result_card table { width: 100%; }
.st-key-placeholder { align-items: center; padding: 2.5rem 1rem; opacity: 0.75; }
.st-key-placeholder [data-testid="stMarkdownContainer"],
.st-key-placeholder [data-testid="stCaptionContainer"] { text-align: center; }
.st-key-placeholder .big { font-size: 3rem; line-height: 1; margin-bottom: 0.75rem; }
/* 업로드 칸을 "여기에 놓으세요" 영역처럼 보이게 한다 (칸 전체가 클릭·드롭 영역) */
[data-testid="stFileUploaderDropzone"] {
    min-height: 7rem; border: 2px dashed var(--primary-color, #B23A2E);
    justify-content: center; cursor: pointer; transition: background-color 0.15s;
}
[data-testid="stFileUploaderDropzone"]:hover { background-color: rgba(178, 58, 46, 0.06); }
/* 미리보기가 세로로 긴 사진이어도 화면을 다 차지하지 않게 한다 */
.st-key-photo_panel [data-testid="stImage"] img { max-height: 45vh; object-fit: contain; }
/* 넓은 화면: 긴 결과를 스크롤해도 왼쪽 사진이 보이도록 고정한다 */
@media (min-width: 641px) {
    [data-testid="stColumn"]:has(.st-key-photo_panel) {
        position: sticky; top: 4rem; align-self: flex-start;
    }
}
/* 모바일: 사용 방법 카드를 한 줄짜리로 줄여 업로드 영역이 먼저 보이게 한다 */
@media (max-width: 640px) {
    .st-key-steps [data-testid="stCaptionContainer"] { display: none; }
    .st-key-steps [data-testid="stVerticalBlockBorderWrapper"],
    .st-key-steps [data-testid="stVerticalBlock"] { padding-top: 0.35rem; padding-bottom: 0.35rem; gap: 0.5rem; }
    .st-key-hero h1 { font-size: 2rem; }
}
</style>
"""

STEPS = [
    ("📸", "사진 올리기", "정면 얼굴 사진을 올리거나 촬영해요"),
    ("🔮", "관상 보기", "버튼을 누르면 관상가가 얼굴을 살펴봐요"),
    ("📜", "결과 확인", "부위별 풀이와 운세를 확인해요"),
]


def render_sidebar() -> None:
    with st.sidebar:
        st.header("🔮 AI 관상가")
        st.write(
            "AI가 전통 관상학을 바탕으로 이마, 눈, 코, 입, 귀, 턱의 생김새를 "
            "풀이해 드리는 재미용 서비스입니다."
        )
        st.subheader("📷 좋은 사진 고르는 팁")
        st.markdown(
            "- 얼굴 **정면**이 잘 보이는 사진\n"
            "- **밝은 조명**, 그림자가 적은 사진\n"
            "- 모자·선글라스·마스크는 벗고\n"
            "- 한 사람만 나온 사진"
        )
        st.subheader("🔒 개인정보 안내")
        st.caption(
            "사진은 관상 풀이를 위해 OpenAI로 전송되며, 이 앱의 서버에는 저장되지 않습니다. "
            "다른 사람의 사진은 본인의 동의를 받고 사용해주세요."
        )


def render_header() -> None:
    with st.container(key="hero"):
        st.markdown('<span class="seal">觀相</span>', unsafe_allow_html=True)
        st.title("🔮 AI 관상가")
        st.write("얼굴 사진 한 장이면, 40년 경력 관상가가 당신의 얼굴에 담긴 운을 풀어드립니다.")

    with st.container(key="steps"):
        for col, (icon, title, desc) in zip(st.columns(3), STEPS):
            with col.container(border=True):
                st.markdown(f"**{icon} {title}**")
                st.caption(desc)


def render_placeholder() -> None:
    with st.container(key="placeholder"):
        st.markdown('<div class="big">📜</div>', unsafe_allow_html=True)
        st.markdown("**아직 풀이한 관상이 없습니다**")
        st.caption("왼쪽에서 사진을 올리고 「관상 보기」를 눌러주세요.")


def main() -> None:
    st.set_page_config(
        page_title="AI 관상가",
        page_icon="🔮",
        layout="wide",
        initial_sidebar_state="auto",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    api_key = get_api_key()
    model = get_model()

    render_sidebar()
    render_header()

    left, right = st.columns([5, 6], gap="large")

    with left, st.container(key="photo_panel"):
        st.subheader("사진")
        data = get_input_bytes()

        # 새 이미지가 들어오면 이전 결과를 초기화한다.
        image_id = hashlib.sha256(data).hexdigest() if data else None
        if st.session_state.get("image_id") != image_id:
            st.session_state.image_id = image_id
            st.session_state.result = None

        image = load_image(data) if data is not None else None
        if image is not None:
            st.image(image, caption="분석할 사진", width="stretch")
        clicked = st.button(
            "🔮 관상 보기",
            type="primary",
            width="stretch",
            disabled=image is None,
        )

    with right:
        st.subheader("관상 풀이")
        with st.container(border=True, key="result_card"):
            if clicked and image is not None:
                with st.spinner("관상가가 얼굴을 살펴보는 중입니다..."):
                    st.session_state.result = run_analysis(image, api_key, model)

            if st.session_state.get("result"):
                render_result(st.session_state.result)
            elif not clicked:
                render_placeholder()

    st.divider()
    st.caption(f"⚠️ {DISCLAIMER}")


if __name__ == "__main__":
    main()
