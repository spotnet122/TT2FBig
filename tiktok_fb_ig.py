#!/usr/bin/env python3
"""
TikTok -> Facebook Page (+ Instagram if linked) auto-poster.

Flow:
  1. Round-robin across TIKTOK_PROFILES; for the current profile, fetch a
     small page of videos via yt-dlp starting at that profile's saved
     cursor (never the whole list — cheap even for huge profiles).
  2. Dedupe against state.json (by tiktok video_id). Once a page is fully
     posted, the cursor moves past it, so nothing gets re-checked or
     skipped. Once a profile's list is confirmed exhausted (hasMore=false
     with nothing new), move to the next profile and reset its cursor.
  3. Download the (no-watermark) video.
  4. Groq rewrites the TikTok caption into a fresh, niche-agnostic caption
     (works for any content type — news, devotional, motivation, clips, etc.)
  5. Upload video to the FB Page (system user token -> page token exchange).
  6. If the Page has a linked Instagram Business Account, upload the video
     to uguu.se (temp public host, ~24-48h) to get a public URL, then post
     it as an IG Reel via that URL. If no IG account is linked -> skip IG.
  7. Save dedupe/cursor state.

ENV VARS (put in .env or GitHub Actions secrets):
  FB_SYSTEM_USER_TOKEN  - Meta system user access token
  FB_PAGE_ID            - target Facebook Page ID
  GROQ_API_KEY          - Groq API key
  GROQ_MODEL            - optional, default "llama-3.3-70b-versatile"
  MAX_VIDEOS            - optional, default 1 (videos posted per run)
  STATE_FILE            - optional, default "state.json"

TikTok profiles are NOT a secret — edit TIKTOK_PROFILES below directly in
code (git tracks the history, so old ids never silently disappear like
they do when you overwrite a GitHub secret).

Usage:
  python tiktok_fb_ig_pipeline.py --dry-run   # test without posting
  python tiktok_fb_ig_pipeline.py             # live run
"""

import os
import sys
import json
import time
import argparse
import pathlib
import requests
import yt_dlp

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# TikTok source profiles, no "@". Round-robin: pipeline stays on a profile
# until ALL of its videos are posted, then automatically moves to the next
# one in this list (wrapping back to the top). Add/remove/reorder freely —
# just edit this list and commit.
TIKTOK_PROFILES = [
    "storyflix11",
]

FB_SYSTEM_USER_TOKEN = os.environ.get("FB_SYSTEM_USER_TOKEN", "")
FB_PAGE_ID = os.environ.get("FB_PAGE_ID", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
MAX_VIDEOS = int(os.environ.get("MAX_VIDEOS", "1"))
DEDUPE_DIR = pathlib.Path(os.environ.get("DEDUPE_DIR", "dedupe"))
STATE_FILE = DEDUPE_DIR / os.environ.get("STATE_FILE", "state.json")
GRAPH_VERSION = "v20.0"
TMP_DIR = pathlib.Path("tmp_videos")
PAGE_SIZE = 5          # how many videos to check at a time (lightweight)
MAX_PAGES_PER_RUN = 5  # safety bound: at most PAGE_SIZE*MAX_PAGES_PER_RUN videos touched per profile per run

REQUIRED = {
    "FB_SYSTEM_USER_TOKEN": FB_SYSTEM_USER_TOKEN,
    "FB_PAGE_ID": FB_PAGE_ID,
    "GROQ_API_KEY": GROQ_API_KEY,
}


def check_env():
    missing = [k for k, v in REQUIRED.items() if not v]
    if missing:
        print(f"[FATAL] Missing env vars: {', '.join(missing)}")
        sys.exit(1)
    if not TIKTOK_PROFILES:
        print("[FATAL] TIKTOK_PROFILES list is empty — add at least one profile in the code.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Dedupe / round-robin state
#
# Each profile remembers its own `cursor` (position in its video list) and
# `posted_ids`. Every run only fetches a small page starting at that saved
# cursor — never the whole list — so dedupe checking stays cheap even for
# profiles with thousands of videos. The cursor only moves forward once a
# page is confirmed fully posted, which is what guarantees nothing gets
# skipped.
# ---------------------------------------------------------------------------
def load_state():
    DEDUPE_DIR.mkdir(parents=True, exist_ok=True)  # auto-create dedupe folder
    if STATE_FILE.exists():
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            state = json.load(f)
    else:
        state = {}
    state.setdefault("current_profile_index", 0)
    state.setdefault("profiles", {})  # { "profile_name": {"cursor": int, "posted_ids": [...]} }

    # migrate from the older flat "posted_ids": {profile: [ids]} schema so
    # nothing gets lost / re-posted when upgrading.
    old_posted = state.pop("posted_ids", None)
    if old_posted:
        for p, ids in old_posted.items():
            state["profiles"].setdefault(p, {"cursor": 0, "posted_ids": []})
            existing = set(state["profiles"][p]["posted_ids"])
            state["profiles"][p]["posted_ids"] = list(existing | set(ids))

    for p in TIKTOK_PROFILES:
        state["profiles"].setdefault(p, {"cursor": 0, "posted_ids": []})

    return state


def save_state(state):
    DEDUPE_DIR.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


# ---------------------------------------------------------------------------
# TikTok fetch (yt-dlp — TikWM's API started returning 403 on GitHub Actions
# IPs, so listing/caption/download all go through yt-dlp directly against
# tiktok.com instead)
# ---------------------------------------------------------------------------
def _ydl_opts(**extra):
    opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        # a normal-looking UA helps avoid TikTok's basic bot filtering
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            )
        },
    }
    opts.update(extra)
    return opts


def list_profile_videos(profile):
    """Cheap flat listing of ALL video ids/urls for a profile (no per-video
    metadata download). Order matches the profile's public feed (newest
    first), which is what the `cursor` indexes into."""
    url = f"https://www.tiktok.com/@{profile}"
    with yt_dlp.YoutubeDL(_ydl_opts(extract_flat=True)) as ydl:
        info = ydl.extract_info(url, download=False)
    entries = info.get("entries") or []
    videos = []
    for e in entries:
        vid = e.get("id")
        if not vid:
            continue
        videos.append({
            "id": vid,
            "url": e.get("url") or e.get("webpage_url") or f"https://www.tiktok.com/@{profile}/video/{vid}",
        })
    return videos


def fetch_video_meta(video_url):
    """Full metadata for ONE video (needed for the caption) — only called
    for videos we're actually about to consider posting, to keep this
    cheap."""
    with yt_dlp.YoutubeDL(_ydl_opts()) as ydl:
        info = ydl.extract_info(video_url, download=False)
    caption = (info.get("description") or info.get("title") or "").strip()
    return {"caption": caption}


def find_candidates(profile, pstate):
    """Walk forward from the profile's saved cursor, page by page (bounded
    by MAX_PAGES_PER_RUN), until it finds un-posted videos or confirms the
    profile is exhausted. Advances/saves pstate['cursor'] as it goes, so
    fully-posted pages are never re-scanned on a later run.

    The full (flat, cheap) video list is fetched once per profile per run;
    only videos that turn out to be un-posted get a full per-video
    metadata fetch (for the caption)."""
    all_videos = list_profile_videos(profile)
    if not all_videos:
        return [], True  # profile empty / unreachable -> treat as exhausted

    total = len(all_videos)
    cursor = pstate["cursor"]
    posted = set(pstate["posted_ids"])

    for _ in range(MAX_PAGES_PER_RUN):
        if cursor >= total:
            pstate["cursor"] = cursor
            return [], True  # exhausted

        page = all_videos[cursor:cursor + PAGE_SIZE]
        unposted = [v for v in page if v["id"] not in posted]

        if unposted:
            pstate["cursor"] = cursor  # don't advance past this page yet
            candidates = []
            for v in unposted:
                meta = fetch_video_meta(v["url"])
                candidates.append({"id": v["id"], "caption": meta["caption"], "url": v["url"]})
            return candidates, False  # found new videos, not exhausted

        # this whole page was already posted -> safe to move the cursor
        # past it and check the next page
        cursor += len(page)
        pstate["cursor"] = cursor
        if cursor >= total:
            return [], True  # exhausted

    # hit the per-run page bound without resolving -> not exhausted, just
    # deferred to the next run (cursor already saved at furthest point reached)
    return [], False


def download_video(video_url, dest_path):
    """Download the highest-quality (HD, no watermark) version of a TikTok
    video via yt-dlp straight to dest_path."""
    opts = _ydl_opts(
        skip_download=False,
        outtmpl=str(dest_path),
        format="bestvideo*+bestaudio/best",
        format_sort=["res", "fps", "br"],
        merge_output_format="mp4",
    )
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([video_url])
    return dest_path


# ---------------------------------------------------------------------------
# Facebook: system user token -> page token, + IG account lookup
# ---------------------------------------------------------------------------
def get_page_token_and_ig(system_token, page_id):
    url = f"https://graph.facebook.com/{GRAPH_VERSION}/me/accounts"
    params = {"access_token": system_token, "limit": 200}
    page_token = None
    while url:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        for page in data.get("data", []):
            if page["id"] == page_id:
                page_token = page["access_token"]
                break
        if page_token:
            break
        paging = data.get("paging", {})
        url = paging.get("next")
        params = {}  # next url already has query params
    if not page_token:
        raise RuntimeError(f"Page {page_id} not found under this system user token")

    # Check for linked Instagram Business Account
    ig_id = None
    r = requests.get(
        f"https://graph.facebook.com/{GRAPH_VERSION}/{page_id}",
        params={"fields": "instagram_business_account", "access_token": page_token},
        timeout=30,
    )
    if r.ok:
        ig_id = r.json().get("instagram_business_account", {}).get("id")

    return page_token, ig_id


# ---------------------------------------------------------------------------
# Groq: niche-agnostic caption rewrite
# ---------------------------------------------------------------------------
def generate_caption(original_caption, groq_key, model=GROQ_MODEL, max_words=15, max_attempts=3):
    base_prompt = (
        "You are a social media caption writer. You will be given the ORIGINAL "
        "caption from a TikTok video (any topic/niche — news, devotional, motivation, "
        "comedy, film clip, etc). Rewrite it for Facebook/Instagram in the SAME "
        "language/tone as the original (if it's Hindi/Hinglish, reply in Hindi/Hinglish; "
        "if English, reply in English). Output in this EXACT format:\n\n"
        "<one punchy caption sentence>\n"
        "\n"
        "#hashtag1\n#hashtag2\n#hashtag3\n#hashtag4\n#hashtag5\n\n"
        "Strict rules:\n"
        f"- Line 1 is ONE sentence, MAXIMUM {max_words} words (count carefully — "
        "9 or 10 words is ideal, never go over the limit).\n"
        "- No emojis, no digits/numbers, in line 1.\n"
        "- Do NOT invent facts not present in the original caption.\n"
        "- Do NOT mention TikTok, any username/handle, video ID, or any TikTok-related "
        "word anywhere — the sentence must read naturally with no gaps or leftover "
        "words if such a mention were ever removed.\n"
        "- Then a blank line, then EXACTLY 5 hashtags, one per line.\n"
        "- Hashtags must be SEO-relevant to the actual content/niche of this video "
        "(not generic filler like #viral #trending #fyp) — think what someone would "
        "actually search to find this content. No spaces, no numbers, no repeats, "
        "no # symbol duplicated.\n"
        "- Output ONLY the caption + hashtags, nothing else (no preamble, no quotes, "
        "no explanation).\n\n"
        f"ORIGINAL CAPTION:\n{original_caption or '(no caption provided)'}"
    )

    prompt = base_prompt
    body_raw, hashtag_lines = "", []

    for attempt in range(1, max_attempts + 1):
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {groq_key}"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 150,
            },
            timeout=30,
        )
        r.raise_for_status()
        raw = r.json()["choices"][0]["message"]["content"].strip()
        body_raw, hashtag_lines = split_caption(raw)

        if has_forbidden_terms(body_raw):
            # don't strip mid-sentence and leave gaps -> ask the model to
            # rewrite it cleanly instead
            prompt = (
                base_prompt
                + "\n\nYour previous attempt mentioned TikTok, a video ID, a handle, "
                  "or a number in line 1. Rewrite line 1 from scratch as a natural "
                  "sentence with NO such mentions and no leftover gaps."
            )
            continue

        body = clean_body(body_raw)
        if body and len(body.split()) <= max_words:
            break

        prompt = (
            base_prompt
            + f"\n\nYour previous attempt was too long. Rewrite line 1 to be "
              f"{max_words} words or fewer, same meaning, same format."
        )
    else:
        # last resort after all retries: strip forbidden terms, trim to
        # max_words, and tidy up any dangling words left behind
        body = clean_body(body_raw)
        words = body.split()
        if len(words) > max_words:
            words = words[:max_words]
        body = tidy_dangling(" ".join(words))

    hashtags = clean_hashtags(hashtag_lines)
    if not hashtags:
        hashtags = fallback_hashtags(original_caption)

    return f"{body}\n\n" + "\n".join(hashtags)


def has_forbidden_terms(text):
    """True if the raw caption line still mentions TikTok/handles/IDs/numbers."""
    import re

    if re.search(r"(?i)\btik\s?tok\b", text):
        return True
    if re.search(r"(?i)\b(video|clip|post|reel)?\s*id\b", text):
        return True
    if re.search(r"@\w+", text):
        return True
    if re.search(r"\d", text):
        return True
    return False


def split_caption(raw):
    """Split raw model output into (body_line_raw, hashtag_lines) without
    cleaning yet, so forbidden-term detection sees the untouched text."""
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    body_lines, hashtag_lines = [], []
    for l in lines:
        if l.startswith("#"):
            hashtag_lines.append(l)
        else:
            body_lines.append(l)
    return " ".join(body_lines).strip(), hashtag_lines


def clean_body(body_raw):
    """Strip tiktok/id/handle/digit mentions and collapse to one sentence."""
    import re

    text = re.sub(r"(?i)\btik\s?tok\b", "", body_raw)
    text = re.sub(r"(?i)\b(video|clip|post|reel)?\s*id\b", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\s+([.,!?])", r"\1", text).strip()

    # keep only the first sentence if the model gave more than one
    match = re.search(r"[^.!?]+[.!?]?", text)
    if match:
        text = match.group(0).strip()
    return text


def tidy_dangling(text):
    """Remove leftover leading/trailing function words (e.g. 'from', 'by')
    that can get orphaned after stripping a mention, and fix capitalization
    / end punctuation."""
    edge_words = {"by", "from", "on", "in", "at", "with", "to", "of", "the",
                  "a", "an", "and", "or", "this", "that", "for"}
    words = text.split()
    while words and words[-1].strip(".,!?").lower() in edge_words:
        words.pop()
    while words and words[0].lower() in edge_words:
        words.pop(0)
    text = " ".join(words).strip()
    if text:
        text = text[0].upper() + text[1:]
        if text[-1] not in ".!?":
            text += "."
    return text


def clean_hashtags(hashtag_lines):
    import re

    hashtags = []
    for l in hashtag_lines:
        hashtags.extend(re.findall(r"#\w+", l))

    seen, clean_tags = set(), []
    for h in hashtags:
        h = re.sub(r"[^A-Za-z#]", "", h)
        if len(h) > 1 and h.lower() not in seen:
            seen.add(h.lower())
            clean_tags.append(h)
    return clean_tags[:5]


def fallback_hashtags(original_caption):
    """Only used if the model fails to return usable hashtags at all —
    pulls keywords from the original caption instead of generic filler."""
    import re

    words = re.findall(r"[A-Za-z]{4,}", original_caption or "")
    stop = {"this", "that", "with", "from", "your", "have", "will", "just",
            "what", "when", "they", "them", "video", "watch", "tiktok"}
    seen, tags = set(), []
    for w in words:
        wl = w.lower()
        if wl in stop or wl in seen:
            continue
        seen.add(wl)
        tags.append("#" + w.capitalize())
        if len(tags) == 5:
            break
    generic_pool = ["#Explore", "#DailyDose", "#MustWatch", "#Highlights", "#ForYou"]
    for g in generic_pool:
        if len(tags) >= 5:
            break
        if g.lower() not in seen:
            seen.add(g.lower())
            tags.append(g)
    return tags[:5]


# ---------------------------------------------------------------------------
# Facebook upload
# ---------------------------------------------------------------------------
def upload_to_facebook(page_id, page_token, video_path, caption):
    url = f"https://graph-video.facebook.com/{GRAPH_VERSION}/{page_id}/videos"
    with open(video_path, "rb") as f:
        files = {"source": f}
        data = {"description": caption, "access_token": page_token}
        r = requests.post(url, files=files, data=data, timeout=600)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# uguu.se temp host (needed because IG Graph API requires a public video_url)
# ---------------------------------------------------------------------------
def upload_to_uguu(video_path):
    with open(video_path, "rb") as f:
        files = {"files[]": f}
        r = requests.post("https://uguu.se/upload", files=files, timeout=600)
    r.raise_for_status()
    data = r.json()
    # response: {"success": true, "files": [{"url": "...", ...}]}
    return data["files"][0]["url"]


# ---------------------------------------------------------------------------
# Instagram Reels post (via public video_url)
# ---------------------------------------------------------------------------
def post_to_instagram(ig_id, page_token, video_url, caption):
    # 1. Create media container
    r = requests.post(
        f"https://graph.facebook.com/{GRAPH_VERSION}/{ig_id}/media",
        data={
            "media_type": "REELS",
            "video_url": video_url,
            "caption": caption,
            "access_token": page_token,
        },
        timeout=60,
    )
    r.raise_for_status()
    creation_id = r.json()["id"]

    # 2. Poll until video finishes processing
    status_url = f"https://graph.facebook.com/{GRAPH_VERSION}/{creation_id}"
    for _ in range(30):  # up to ~5 min
        s = requests.get(
            status_url,
            params={"fields": "status_code", "access_token": page_token},
            timeout=30,
        )
        s.raise_for_status()
        status = s.json().get("status_code")
        if status == "FINISHED":
            break
        if status == "ERROR":
            raise RuntimeError("IG media processing failed")
        time.sleep(10)
    else:
        raise RuntimeError("IG media processing timed out")

    # 3. Publish
    p = requests.post(
        f"https://graph.facebook.com/{GRAPH_VERSION}/{ig_id}/media_publish",
        data={"creation_id": creation_id, "access_token": page_token},
        timeout=60,
    )
    p.raise_for_status()
    return p.json()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Fetch/generate only, don't post")
    args = parser.parse_args()

    check_env()
    TMP_DIR.mkdir(exist_ok=True)

    state = load_state()
    n = len(TIKTOK_PROFILES)
    start_idx = state["current_profile_index"] % n

    # Walk profiles round-robin starting where we left off. Each profile is
    # checked a small page at a time from its saved cursor — a profile is
    # only skipped once that walk confirms nothing is left (hasMore=false),
    # which is what guarantees no video gets missed.
    profile = None
    candidates = []
    for step in range(n):
        idx = (start_idx + step) % n
        p = TIKTOK_PROFILES[idx]
        pstate = state["profiles"][p]
        print(f"Checking @{p} from cursor {pstate['cursor']} ({step + 1}/{n})")
        c, exhausted = find_candidates(p, pstate)

        if c:
            profile, candidates = p, c
            state["current_profile_index"] = idx
            print(f"  {len(c)} new video(s) found")
            break
        elif exhausted:
            print(f"  @{p} fully exhausted, resetting cursor, moving to next profile")
            pstate["cursor"] = 0
        else:
            print(f"  no new videos found in this run's page budget, will resume next run")

    save_state(state)  # persist any cursor progress even if nothing was postable this run

    if not candidates:
        print("\nNo postable video this run. Exiting.")
        return

    print("Fetching Facebook page token")
    page_token, ig_id = get_page_token_and_ig(FB_SYSTEM_USER_TOKEN, FB_PAGE_ID)
    print(f"Instagram: {'linked' if ig_id else 'not linked, skipping'}")

    # Post exactly ONE video per run, trying the next candidate if one fails.
    posted = False
    for v in candidates:
        vid, tt_caption = v["id"], v["caption"]
        video_path = TMP_DIR / f"{vid}.mp4"
        print(f"\nVideo {vid} (@{profile})")

        try:
            download_video(v["url"], video_path)

            caption = generate_caption(tt_caption, GROQ_API_KEY)
            print(f"Caption: {caption}")

            if args.dry_run:
                print("Dry run, not posting.")
            else:
                fb_res = upload_to_facebook(FB_PAGE_ID, page_token, video_path, caption)
                print(f"Posted to Facebook: {fb_res.get('id')}")

                if ig_id:
                    public_url = upload_to_uguu(video_path)
                    ig_res = post_to_instagram(ig_id, page_token, public_url, caption)
                    print(f"Posted to Instagram: {ig_res.get('id')}")

            # mark as posted only after success (or dry-run, to avoid re-testing same video)
            state["profiles"][profile]["posted_ids"].append(vid)
            save_state(state)
            posted = True

        except Exception as e:
            print(f"Failed: {e}")
            # do NOT mark as posted -> will retry this video next run
        finally:
            if video_path.exists():
                video_path.unlink()

        if posted:
            break

    if not posted:
        print("\nNo video posted this run (all candidates failed).")

    # cleanup: remove tmp dir if empty
    try:
        TMP_DIR.rmdir()
    except OSError:
        pass

    print("Done.")


if __name__ == "__main__":
    main()
