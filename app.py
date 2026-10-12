
import os
import tempfile
import time

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv(override=True)

MODEL = "gemini-3.5-flash-lite"  # Current model; supports Google Search grounding
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
7. Distinguish between no evidence being found and evidence showing a claim is false.
8. Label sources as retrieved, independently checked, or unverified.
9. Do not treat sensational language, AI suspicion, or lack of search
   results as proof that a claim is false.
10. A real image may be paired with a false caption. A genuine voice
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

Confidence must describe confidence in the VERDICT based on the quality,
relevance, recency, and independence of evidence—not how plausible the claim
sounds. Use this rubric consistently:
- HIGH: at least two credible, independent, recent sources directly support
  the core claim, or a strong primary/official source conclusively supports it;
  no material contradiction remains unresolved.
- MEDIUM: one credible source directly supports the claim, or multiple sources
  support only part of it, with some details still unconfirmed.
- LOW: no usable sources were retrieved, sources are weak/unverified, evidence
  conflicts materially, or the claim cannot be evaluated with available context.
Do not automatically assign Low confidence just because the claim concerns a
celebrity death, health event, or breaking news. Evaluate the actual retrieved
sources and their publication dates. For multi-part headlines, assess each
part separately (for example, death, age, and circumstances) and let the final
verdict reflect the least-supported material part. If reliable sources directly
confirm the core claim, choose Supported rather than Insufficient Evidence.
For claims containing words like "today", "here", "now", or "near me", identify
missing location/date/time context when it materially affects verification.
If web search failed or no sources were retrieved, state that external
verification was not completed and keep confidence Low. Do not say that no
relevant evidence exists merely because no search sources were returned.
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


def _read_field(obj, *names):
    """Read SDK/Pydantic attributes or dictionary keys across naming styles."""
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj.get(name)
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return None


def extract_sources(response):
    """Extract deduplicated web URLs from Gemini Search grounding metadata."""
    found = []
    seen = set()

    try:
        candidates = _read_field(response, "candidates") or []
        for candidate in candidates:
            metadata = _read_field(candidate, "grounding_metadata", "groundingMetadata")
            if not metadata:
                continue

            chunks = _read_field(metadata, "grounding_chunks", "groundingChunks") or []
            for chunk in chunks:
                web_info = _read_field(chunk, "web")
                if not web_info:
                    continue

                url = _read_field(web_info, "uri", "url")
                title = _read_field(web_info, "title") or url
                if isinstance(url, str) and url.startswith(("https://", "http://")) and url not in seen:
                    found.append({"title": str(title), "url": url})
                    seen.add(url)
    except Exception as exc:
        # Do not fail the entire analysis if metadata has an unexpected shape.
        st.caption(f"Could not parse grounding metadata: {type(exc).__name__}")

    return found


def grounding_diagnostics(response):
    """Expose whether the API returned grounding metadata, without showing secrets."""
    result = {"candidate_count": 0, "metadata_found": False, "search_queries": [], "chunk_count": 0}
    try:
        candidates = _read_field(response, "candidates") or []
        result["candidate_count"] = len(candidates)
        for candidate in candidates:
            metadata = _read_field(candidate, "grounding_metadata", "groundingMetadata")
            if not metadata:
                continue
            result["metadata_found"] = True
            queries = _read_field(metadata, "web_search_queries", "webSearchQueries") or []
            result["search_queries"].extend(str(q) for q in queries)
            chunks = _read_field(metadata, "grounding_chunks", "groundingChunks") or []
            result["chunk_count"] += len(chunks)
    except Exception:
        pass
    result["search_queries"] = list(dict.fromkeys(result["search_queries"]))
    return result


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


def _call_model_with_retry(client, *, contents, config):
    """Retry transient model-capacity errors a small number of times."""
    delays = (2, 5)
    for attempt in range(len(delays) + 1):
        try:
            return client.models.generate_content(
                model=MODEL,
                contents=contents,
                config=config,
            )
        except Exception as exc:
            message = str(exc).upper()
            transient = (
                "503" in message
                or "UNAVAILABLE" in message
                or "INTERNAL" in message
                or "502" in message
                or "504" in message
            )
            if not transient or attempt >= len(delays):
                raise
            st.warning(
                f"Gemini is temporarily busy. Retrying "
                f"({attempt + 1}/{len(delays)})..."
            )
            time.sleep(delays[attempt])


def _generate_with_fallback(client, contents):
    """Try grounded web search first; if that tool's quota is unavailable, try without it."""
    try:
        return _call_model_with_retry(
            client,
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

        # Do not silently hide that evidence gathering failed. A plain generation
        # fallback can produce text, but it cannot count as source-based verification.
        st.warning(
            "Google Search grounding hit a quota/rate-limit error. VERITAS is "
            "falling back to a non-grounded response; this run may end as "
            "Insufficient Evidence. Check your Google AI Studio project and grounding quota."
        )
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

        return _call_model_with_retry(
            client,
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
    grounding_info = grounding_diagnostics(first)

    # If Gemini returns no Search grounding sources, make one focused retry.
    # Grounding is automatic, so this retry improves coverage but cannot
    # guarantee that the provider will return sources on every request.
    if not sources:
        st.warning(
            "The first analysis returned no source URLs. VERITAS is trying one "
            "focused grounded request. If your API tier does not allow Search "
            "grounding, the app cannot verify current claims from this model call."
        )
        focused_prompt = prompt + """

IMPORTANT SEARCH RETRY:
The first attempt returned no source URLs. Re-examine the main factual claim
and make a focused Google Search grounding attempt, prioritizing recent credible
news outlets and primary/official sources. If the claim concerns a person's
reported death or hospitalization, check the date carefully and seek multiple
independent sources. Do not infer that a claim is false or untrue just because
search results are absent. If this attempt still returns no sources, explicitly state that Search grounding
returned no source URLs and that external verification could not be completed.
Do not fabricate sources or claim the search confirmed the claim.
"""
        try:
            retry_response = _generate(
                client,
                focused_prompt,
                content=content,
                mime_type=mime_type,
                video_path=video_path,
            )
            retry_sources = extract_sources(retry_response)
            if retry_sources:
                first = retry_response
                initial_report = first.text or initial_report
                sources = retry_sources
                grounding_info = grounding_diagnostics(retry_response)
        except Exception as retry_exc:
            # Preserve the first useful report if the retry is unavailable.
            st.caption(
                "Focused search retry could not complete; the report below "
                "uses the first response and should be treated as unverified."
            )

    # Give the reviewer the actual URLs extracted from grounding metadata.
    # Previously, the reviewer saw only the prose report, so it could discard
    # or overlook the URLs returned by Search grounding.
    source_context = "\n\nURLs actually returned by Google Search grounding metadata:\n"
    if sources:
        source_context += "\n".join(
            f"- {item['title']}: {item['url']}" for item in sources
        )
    else:
        source_context += "None. Search grounding returned no extractable source URLs."

    # If no sources were returned, do not let a review-only model call create
    # the impression that a second agent independently searched the web.
    # It may still review reasoning, but the report must preserve this limitation.
    try:
        review = _call_model_with_retry(
            client,
            contents=REVIEW_PROMPT + initial_report + source_context,
            config=types.GenerateContentConfig(temperature=0.1),
        )
        final_report = review.text or initial_report
    except Exception:
        final_report = initial_report

    if not sources:
        # Evidence gate: a model-written verdict is not enough when the app has
        # no retrievable citations to show the user.
        limitation = (
            "\n\n### VERITAS evidence gate\n\n"
            "**Final app status: Insufficient Evidence — Low Confidence.**\n\n"
            "No source URLs were extracted from Google Search grounding metadata, "
            "so this run did not complete source-based external verification. "
            "This does not mean the claim is false; it means the claim remains "
            "unverified in this run. The reviewer checks the first report and does "
            "not perform an independent search. Retry later or verify using reliable "
            "primary sources and reputable independent reporting."
        )
        if "veritas evidence gate" not in final_report.lower():
            final_report += limitation
    else:
        # Append the real extracted URLs so users can open and verify them even
        # if the model omits source links from its narrative.
        final_report += "\n\n### Sources extracted from grounding metadata\n" + "\n".join(
            f"- [{item['title']}]({item['url']})" for item in sources
        )

    return final_report, sources, grounding_info

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
    with st.expander("Add context (optional, helpful for claims about today or here)"):
        claim_location = st.text_input("Location", placeholder="City, region, or country")
        claim_datetime = st.text_input("Date and time", placeholder="Date, local time, and timezone if known")
        claim_context = st.text_area("Additional context", placeholder="Where the claim came from, relevant details, etc.", height=80)
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
                contextual_claim = text_input
                context_parts = []
                if claim_location.strip():
                    context_parts.append(f"Location supplied by user: {claim_location.strip()}")
                if claim_datetime.strip():
                    context_parts.append(f"Date/time supplied by user: {claim_datetime.strip()}")
                if claim_context.strip():
                    context_parts.append(f"Additional context supplied by user: {claim_context.strip()}")
                if context_parts:
                    contextual_claim += "\n\nContext supplied by the user (treat as context, not independently verified facts):\n" + "\n".join(context_parts)
                report, sources, grounding_info = analyze(contextual_claim)

            elif mode == "Image":
                report, sources, grounding_info = analyze(
                    uploaded.getvalue(),
                    mime_type=uploaded.type or "image/jpeg",
                )

            elif mode == "Audio":
                report, sources, grounding_info = analyze(
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

                    report, sources, grounding_info = analyze(
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
            st.session_state["grounding_info"] = grounding_info
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
            elif "503" in upper_error or "UNAVAILABLE" in upper_error:
                st.error(
                    "Gemini is temporarily unavailable or under high demand. "
                    "The app retried a few times, but the request still failed. "
                    "Wait briefly and try again; this is not evidence that your "
                    "API quota is exhausted."
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
    st.caption(
        "These are URLs returned in Gemini grounding metadata, not an automatic "
        "guarantee that each source supports the claim. Open them and check the "
        "article date, original context, and whether the source directly verifies it."
    )
    sources = st.session_state.get("sources", [])
    info = st.session_state.get("grounding_info", {})
    with st.expander("Search diagnostics (for troubleshooting)"):
        st.write(f"Model: `{MODEL}`")
        st.write(f"Response candidates: {info.get('candidate_count', 0)}")
        st.write(f"Grounding metadata returned: {info.get('metadata_found', False)}")
        st.write(f"Grounding chunks: {info.get('chunk_count', 0)}")
        queries = info.get("search_queries", [])
        st.write("Search queries reported by Gemini:")
        if queries:
            for query in queries:
                st.write(f"- {query}")
        else:
            st.write("None reported by the API in this response.")

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