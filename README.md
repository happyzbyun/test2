# 🔮 AI 관상가

얼굴 사진을 업로드하면 OpenAI 이미지 분석 모델이 관상을 봐주는 Streamlit 앱입니다.
(재미를 위한 서비스이며 과학적 근거는 없습니다.)

## 로컬 실행

```bash
pip install -r requirements.txt
cp .env.example .env   # .env에 OPENAI_API_KEY 입력
streamlit run app.py
```

## 환경변수

| 이름 | 설명 | 기본값 |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI API 키 (필수) | - |
| `OPENAI_MODEL` | 사용할 Vision 모델 | `gpt-5.4-mini` |

## Streamlit Cloud 배포

Streamlit Cloud는 `.env`를 읽지 않습니다. 앱 설정 → **Secrets**에 아래처럼 등록하세요.

```toml
OPENAI_API_KEY = "sk-..."
OPENAI_MODEL = "gpt-5.4-mini"
```
