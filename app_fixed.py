
import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv(override=True)

MODEL = "gemini-3.8-flash"
MAX_BYTES = 20 * 1024 * 1024  # 20 MB MVP limit

st.set_page_config(
    page_title="VERITAS | AI Fact Checker",
    page_icon="🔎",
    layout="wide",
)

SYSTEM_PROMPT = """
You are the evidence-gathering agent of VERITAS, a fake-news
and factual-claim verification system.

Analyze the supplied text, image, audio, or video.

Tasks:
1. Extract the main factual claims. For media, describe relevant
   visible or spoken claims and distinguish observations from facts.
2. Search the web for relevant, independent evidence using the
   available Google Search tool.
3. Prefer primary sources, official records, and reputable reporting.
4. Look for both supporting and contradicting evidence.
5. Distinguish the original claim from a fact-check of a similar claim.
6. Never invent sources, quotations, dates, or URLs.
7. Do not treat sensational language, AI suspicion, or lack of search
   results as proof that a claim is false.
8. A real image may be paired with a false caption. A genuine voice
   may say something false. Do not confuse authenticity with truth.

Return a clear report with:
- Main claims
- Verdict: Supported / Contradicted / Mixed / Insufficient Evidence
- Evidence supporting the claim
- Evidence contradicting the claim
- Sources and their URLs
- Confidence: Low / Medium / High, with a brief explanation
- Limitations and what remains unverified

Use Contradicted only when credible evidence directly conflicts with
the claim. If evidence is weak, missing, outdated, or inconclusive,
choose Insufficient Evidence. Never invent a numerical probability.
"""

REVIEW_PROMPT = """
You are the independent review step in VERITAS.

Review the initial assessment below. Check whether its conclusion
actually follows from the evidence presented. Identify unsupported
inferences, mismatched dates, misleading context, and sources that
do not directly verify the claim.

Do not invent new evidence or claim to have performed a new search.
Keep source URLs exactly as supplied in the initial assessment.
If the evidence does not justify a firm verdict, use
Insufficient Evidence. Separate media authenticity from claim truth.

Return:
1. Final verdict
2. Key reasons
3. Supporting evidence
4. Contradicting evidence
5. Sources and URLs
6. Confidence and limitations

Initial assessment:
"""

def get_client():
    key = (os.getenv("GEMINI_API_KEY") or "").strip().strip('"').strip("'")
    if not key:
        raise ValueError(
            "GEMINI_API_KEY is missing. Add it to your .env file."
        )
    return genai.Client(api_key=key)


def extract_sources(response):
    """Read source URLs returned in Google Search grounding metadata."""
    found = []
    seen = set()

    try:
        candidates = response.candidates or []
        for candidate in candidates:
            metadata = getattr(candidate, "grounding_metadata", None)
            if not metadata:
                continue

            for chunk in (getattr(metadata, "grounding_chunks", None) or []):
                web_info = getattr(chunk, "web", None)
                if not web_info:
                    continue

                url = getattr(web_info, "uri", None)
                title = getattr(web_info, "title", None) or url

                if url and url not in seen:
                    found.append({"title": title, "url": url})
                    seen.add(url)
    except Exception:
        pass

    return found


def _generate(client, prompt, content=None, mime_type=None, video_path=None):
    """Run the evidence request with Search grounding, then fall back if its quota is unavailable."""
    if video_path:
        media_file = client.files.upload(file=video_path)
        try:
            contents = [prompt, media_file]
            return _generate_with_fallback(client, contents)
        finally:
            try:
                client.files.delete(name=media_file.name)
            except Exception:
                pass

    if mime_type:
        part = types.Part.from_bytes(data=content, mime_type=mime_type)
        return _generate_with_fallback(client, [prompt, part])

    return _generate_with_fallback(
        client, prompt + "\n\nUser-submitted claim:\n" + str(content)
    )


def _generate_with_fallback(client, contents):
    """Try grounded web search first; if that tool's quota is unavailable, try without it."""
    try:
        return client.models.generate_content(
            model=MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                temperature=0.1,
            ),
        )
    except Exception as exc:
        message = str(exc).upper()
        quota_error = (
            "429" in message
            or "RESOURCE_EXHAUSTED" in message
            or "QUOTA" in message
        )
        if not quota_error:
            raise

        # The error may be specific to Search grounding. A plain generation
        # request can still work if the model's general generation quota remains.
        fallback_note = (
            "\n\nImportant: Google Search grounding was unavailable. "
            "Do not claim to have searched the web or invent sources. "
            "Clearly mark externally unverified information."
        )
        if isinstance(contents, str):
            fallback_contents = contents + fallback_note
        elif isinstance(contents, list) and contents and isinstance(contents[0], str):
            fallback_contents = [contents[0] + fallback_note, *contents[1:]]
        else:
            fallback_contents = contents

        return client.models.generate_content(
            model=MODEL,
            contents=fallback_contents,
            config=types.GenerateContentConfig(temperature=0.1),
        )


def analyze(content, mime_type=None, video_path=None):
    client = get_client()

    prompt = SYSTEM_PROMPT + """
Analyze the provided content. Search for real evidence when Search grounding
is available. Include source titles and URLs only when actually provided by
grounding metadata or reliably present in the evidence. If Search is unavailable,
say that external verification could not be completed.
"""

    first = _generate(
        client,
        prompt,
        content=content,
        mime_type=mime_type,
        video_path=video_path,
    )

    initial_report = first.text or "No report was returned."
    sources = extract_sources(first)

    # Optional second call: keep it only when the first call succeeded.
    # If review quota is exhausted, preserve the useful initial report.
    try:
        review = client.models.generate_content(
            model=MODEL,
            contents=REVIEW_PROMPT + initial_report,
            config=types.GenerateContentConfig(temperature=0.1),
        )
        final_report = review.text or initial_report
    except Exception:
        final_report = initial_report

    return final_report, sources

st.title("🔎 VERITAS")
st.subheader("Agentic AI Fake News Detection")
st.caption(
    "Verify factual claims across text, images, audio, and video. "
    "Results are evidence-based assessments, not guarantees of truth."
)

with st.sidebar:
    st.header("About VERITAS")
    st.write("**Agent 1 — Evidence Researcher**")
    st.write("Searches for relevant supporting and contradicting evidence.")
    st.write("**Agent 2 — Critical Reviewer**")
    st.write("Challenges the initial assessment and checks for overclaiming.")
    st.divider()
    st.caption("MVP limits: 20 MB per upload.")
    st.caption("Never upload confidential or sensitive media.")

mode = st.radio(
    "Choose input type",
    ["Text", "Image", "Audio", "Video"],
    horizontal=True,
)

text_input = ""
uploaded = None

if mode == "Text":
    text_input = st.text_area(
        "Enter a news claim or headline",
        placeholder="Example: A claim you want to verify...",
        height=150,
    )
elif mode == "Image":
    uploaded = st.file_uploader(
        "Upload an image",
        type=["png", "jpg", "jpeg", "webp"],
    )
elif mode == "Audio":
    uploaded = st.file_uploader(
        "Upload audio",
        type=["wav", "mp3", "m4a", "ogg", "flac", "webm"],
    )
else:
    uploaded = st.file_uploader(
        "Upload a short video",
        type=["mp4", "mov", "webm", "mpeg", "mpg"],
    )

if st.button("Analyze with VERITAS", type="primary"):
    if mode == "Text" and not text_input.strip():
        st.warning("Please enter a claim first.")
        st.stop()

    if mode != "Text" and uploaded is None:
        st.warning("Please upload a file first.")
        st.stop()

    if mode != "Text" and uploaded.size > MAX_BYTES:
        st.error("File exceeds the 20 MB MVP limit.")
        st.stop()

    with st.status("VERITAS is analyzing the submission...", expanded=True) as status:
        try:
            st.write("Step 1: Processing input and gathering evidence...")

            if mode == "Text":
                report, sources = analyze(text_input)

            elif mode == "Image":
                report, sources = analyze(
                    uploaded.getvalue(),
                    mime_type=uploaded.type or "image/jpeg",
                )

            elif mode == "Audio":
                report, sources = analyze(
                    uploaded.getvalue(),
                    mime_type=uploaded.type or "audio/wav",
                )

            else:
                suffix = os.path.splitext(uploaded.name)[1] or ".mp4"
                video_path = None
                try:
                    with tempfile.NamedTemporaryFile(
                        suffix=suffix, delete=False
                    ) as temp:
                        temp.write(uploaded.getvalue())
                        video_path = temp.name

                    report, sources = analyze(
                        content=None,
                        video_path=video_path,
                    )
                finally:
                    if video_path and os.path.exists(video_path):
                        os.remove(video_path)

            st.write("Step 2: Reviewing the evidence and verdict...")
            status.update(
                label="Analysis complete",
                state="complete",
                expanded=False,
            )

            st.session_state["report"] = report
            st.session_state["sources"] = sources
            st.session_state["mode"] = mode

        except Exception as exc:
            status.update(
                label="Analysis could not be completed",
                state="error",
                expanded=True,
            )
            error_text = str(exc)
            upper_error = error_text.upper()

            if "429" in upper_error or "RESOURCE_EXHAUSTED" in upper_error:
                st.error(
                    "Gemini returned a quota/rate-limit error. The app tried "
                    "a fallback request if Search grounding was the cause. "
                    "If this message remains, check the selected model and "
                    "the exact Google AI Studio project/quota."
                )
            elif "401" in upper_error or "UNAUTHENTICATED" in upper_error:
                st.error(
                    "Gemini authentication failed. Check that GEMINI_API_KEY "
                    "contains a valid Gemini API key in your .env file."
                )
            else:
                st.error(
                    "Analysis failed. Check the model name, API access, "
                    "connection, and uploaded file format."
                )

            # Avoid displaying credentials if an SDK error happens to include them.
            safe_error = error_text.replace(
                os.getenv("GEMINI_API_KEY", "__NO_KEY__"), "[REDACTED]"
            )
            st.caption(f"Technical details: {safe_error[:500]}")

if "report" in st.session_state:
    st.divider()
    st.header("Verification Report")
    st.caption(f"Input type: {st.session_state['mode']}")
    st.markdown(st.session_state["report"])

    st.subheader("Sources returned by Google Search")
    sources = st.session_state.get("sources", [])

    if sources:
        for source in sources:
            st.markdown(f"- [{source['title']}]({source['url']})")
    else:
        st.warning(
            "No source URLs were returned in grounding metadata. "
            "Treat any source claims in the report as unverified until "
            "you open and check them independently."
        )

    st.caption(
        "The second agent reviews the first agent's report; it does not "
        "perform an independent search. Always check cited sources."
    )