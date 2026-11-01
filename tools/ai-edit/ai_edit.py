#!/usr/bin/env python3
"""JOYLAB AI 영상 편집 파이프라인 (HyperFrames).

단계:
  transcribe  영상 -> 단어 단위 타임스탬프 (faster-whisper large-v3-turbo)
  plan        무음 구간 + 재테이크(반복 발화) 탐지 -> 유지 구간(plan.json)
  cut         plan.json 대로 잘라 이어붙인 cut.mp4 생성
  subtitles   컷 타임라인 기준 한국어 자막(SRT/JSON), 프로필 규격 적용
  compose     HyperFrames 프로젝트(index.html) 생성: 영상 + 자막 + 효과음
  all         위 전체를 순서대로

필요: ffmpeg, ffprobe, python>=3.9, (transcribe 단계만) faster-whisper
"""
from __future__ import annotations

import argparse
import difflib
import html
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------- utilities


def run(cmd: list[str], capture: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, check=True, text=True, capture_output=capture
    )


def probe_duration(path: Path) -> float:
    out = run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=nw=1:nk=1", str(path),
        ],
        capture=True,
    ).stdout.strip()
    return float(out)


def has_audio(path: Path) -> bool:
    out = run(
        [
            "ffprobe", "-v", "error", "-select_streams", "a",
            "-show_entries", "stream=index", "-of", "csv=p=0", str(path),
        ],
        capture=True,
    ).stdout.strip()
    return bool(out)


def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def save_json(p: Path, data) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------- transcribe


def cmd_transcribe(a) -> None:
    from faster_whisper import WhisperModel  # lazy: heavy import

    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    wav = work / "audio.wav"
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", a.video, "-vn", "-ac", "1",
         "-ar", "16000", str(wav)])

    model = WhisperModel(a.model, device=a.device, compute_type=a.compute_type)
    # vad_filter=False: 재테이크/반복 발화를 보존해야 plan 단계에서 잡을 수 있다.
    # condition_on_previous_text=False: 반복 문장을 "정리"해 버리는 현상 완화.
    segments, info = model.transcribe(
        str(wav),
        language=a.language,
        word_timestamps=True,
        vad_filter=False,
        condition_on_previous_text=False,
    )
    words = []
    for seg in segments:
        for w in seg.words or []:
            t = w.word.strip()
            if t:
                words.append({"text": t, "start": round(w.start, 3), "end": round(w.end, 3)})
    out = {
        "language": info.language,
        "duration": probe_duration(Path(a.video)),
        "model": a.model,
        "words": words,
    }
    save_json(work / "transcript.json", out)
    print(f"[transcribe] {len(words)} words -> {work/'transcript.json'}")


# ---------------------------------------------------------------- plan

_SIL_START = re.compile(r"silence_start:\s*(-?[\d.]+)")
_SIL_END = re.compile(r"silence_end:\s*(-?[\d.]+)")


def detect_silences(media: Path, noise_db: float, min_len: float, duration: float):
    p = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(media), "-vn",
         "-af", f"silencedetect=noise={noise_db}dB:d={min_len}", "-f", "null", "-"],
        text=True, capture_output=True, check=True,
    )
    sil, start = [], None
    for line in p.stderr.splitlines():
        m = _SIL_START.search(line)
        if m:
            start = max(0.0, float(m.group(1)))
            continue
        m = _SIL_END.search(line)
        if m and start is not None:
            sil.append((start, float(m.group(1))))
            start = None
    if start is not None:  # 파일 끝까지 이어진 무음
        sil.append((start, duration))
    return sil


def norm_text(s: str) -> str:
    return re.sub(r"[\s\W_]+", "", s, flags=re.UNICODE).lower()


def group_utterances(words: list[dict], gap: float) -> list[dict]:
    """단어 사이 공백이 gap 이상이면 발화 경계로 본다 ('한 템포 쉬고')."""
    utts: list[dict] = []
    for w in words:
        if utts and w["start"] - utts[-1]["end"] < gap:
            u = utts[-1]
            u["words"].append(w)
            u["end"] = w["end"]
        else:
            utts.append({"start": w["start"], "end": w["end"], "words": [w]})
    for u in utts:
        u["text"] = " ".join(x["text"] for x in u["words"])
        u["norm"] = norm_text(u["text"])
    return utts


def retake_score(a: str, b: str) -> float:
    """a(앞 발화)가 b(뒤 발화)의 재시도인지 0~1 점수.

    - 전체 유사도
    - a 가 b 의 앞부분과 닮았는지 (말하다 끊고 처음부터 다시 한 경우)
    """
    if not a or not b:
        return 0.0
    full = difflib.SequenceMatcher(None, a, b).ratio()
    head = difflib.SequenceMatcher(None, a, b[: max(len(a), int(len(a) * 1.3))]).ratio()
    return max(full, head)


def find_retakes(utts: list[dict], thr: float, lookahead: int, min_chars: int):
    drops = []
    for i, a in enumerate(utts):
        if len(a["norm"]) < min_chars:
            continue
        for j in range(i + 1, min(i + 1 + lookahead, len(utts))):
            b = utts[j]
            sc = retake_score(a["norm"], b["norm"])
            if sc >= thr:
                drops.append(
                    {
                        "index": i,
                        "start": a["start"],
                        "end": a["end"],
                        "resume_at": utts[i + 1]["start"],
                        "text": a["text"],
                        "kept_text": b["text"],
                        "score": round(sc, 3),
                    }
                )
                break
    return drops


def merge_intervals(iv: list[tuple[float, float]]):
    iv = sorted((s, e) for s, e in iv if e > s)
    out: list[list[float]] = []
    for s, e in iv:
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def invert(cuts: list[tuple[float, float]], duration: float):
    keep, cur = [], 0.0
    for s, e in cuts:
        if s > cur:
            keep.append((cur, min(s, duration)))
        cur = max(cur, e)
    if cur < duration:
        keep.append((cur, duration))
    return keep


def build_plan(words, silences, duration, *, retake_gap, retake_thr, lookahead,
               min_chars, keep_pad, min_keep, pre_roll):
    utts = group_utterances(words, retake_gap)
    drops = find_retakes(utts, retake_thr, lookahead, min_chars)

    cuts: list[tuple[float, float]] = []
    # 재테이크: 앞 발화 시작 ~ 다음 발화 시작 직전까지 통째로 삭제
    for d in drops:
        cuts.append((max(0.0, d["start"] - pre_roll), max(0.0, d["resume_at"] - pre_roll)))
    # 무음: 양끝 keep_pad 만 남기고 삭제 (호흡 유지)
    n_sil = 0
    for s, e in silences:
        if e - s > 2 * keep_pad:
            cuts.append((s + keep_pad, e - keep_pad))
            n_sil += 1

    keep = [(s, e) for s, e in invert(merge_intervals(cuts), duration) if e - s >= min_keep]
    kept_dur = sum(e - s for s, e in keep)
    return {
        "duration": round(duration, 3),
        "cut_duration": round(kept_dur, 3),
        "removed": round(duration - kept_dur, 3),
        "silences_cut": n_sil,
        "retakes_dropped": drops,
        "keep": [[round(s, 3), round(e, 3)] for s, e in keep],
    }


def cmd_plan(a) -> None:
    work = Path(a.work)
    tr = load_json(work / "transcript.json")
    dur = tr["duration"]
    sil = detect_silences(Path(a.video), a.noise_db, a.min_silence, dur)
    plan = build_plan(
        tr["words"], sil, dur,
        retake_gap=a.retake_gap, retake_thr=a.retake_thr, lookahead=a.lookahead,
        min_chars=a.min_chars, keep_pad=a.keep_pad, min_keep=a.min_keep,
        pre_roll=a.pre_roll,
    )
    plan["source"] = str(a.video)
    save_json(work / "plan.json", plan)
    print(f"[plan] {plan['duration']}s -> {plan['cut_duration']}s "
          f"(무음 {plan['silences_cut']}곳, 재테이크 {len(plan['retakes_dropped'])}곳 삭제)")
    for d in plan["retakes_dropped"]:
        print(f"  - 삭제 [{d['start']:.2f}-{d['end']:.2f}] \"{d['text']}\"  "
              f"=> 유지 \"{d['kept_text']}\" (유사도 {d['score']})")
    print("  ※ 위 재테이크 목록을 눈으로 확인하세요. 오탐은 --retake-thr 를 올리세요.")


# ---------------------------------------------------------------- cut


def remap_factory(keep: list[list[float]]):
    offsets, acc = [], 0.0
    for s, e in keep:
        offsets.append(acc)
        acc += e - s

    def remap(t: float, with_seg: bool = False):
        for k, ((s, e), off) in enumerate(zip(keep, offsets)):
            if s <= t <= e:
                v = off + (t - s)
                return (v, k) if with_seg else v
        return (None, None) if with_seg else None

    return remap


def build_filter(keep: list[list[float]], with_audio: bool, fade: float = 0.012) -> str:
    parts, labels = [], []
    for i, (s, e) in enumerate(keep):
        d = e - s
        parts.append(f"[0:v]trim=start={s:.3f}:end={e:.3f},setpts=PTS-STARTPTS[v{i}]")
        if with_audio:
            f = min(fade, d / 4)
            parts.append(
                f"[0:a]atrim=start={s:.3f}:end={e:.3f},asetpts=PTS-STARTPTS,"
                f"afade=t=in:d={f:.3f},afade=t=out:st={d - f:.3f}:d={f:.3f}[a{i}]"
            )
            labels.append(f"[v{i}][a{i}]")
        else:
            labels.append(f"[v{i}]")
    n = len(keep)
    if with_audio:
        parts.append("".join(labels) + f"concat=n={n}:v=1:a=1[v][a]")
    else:
        parts.append("".join(labels) + f"concat=n={n}:v=1:a=0[v]")
    return ";\n".join(parts)


def cmd_cut(a) -> None:
    work = Path(a.work)
    plan = load_json(work / "plan.json")
    if not plan["keep"]:
        sys.exit("keep 구간이 비어 있습니다. plan 파라미터를 확인하세요.")
    audio = has_audio(Path(a.video))
    script = work / "filter.txt"
    script.write_text(build_filter(plan["keep"], audio), encoding="utf-8")
    out = work / "cut.mp4"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", a.video,
           "-filter_complex_script", str(script), "-map", "[v]"]
    if audio:
        cmd += ["-map", "[a]", "-c:a", "aac", "-b:a", "192k"]
    cmd += ["-c:v", "libx264", "-crf", "18", "-preset", "medium",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
    run(cmd)
    print(f"[cut] {out}  ({probe_duration(out):.2f}s, 예상 {plan['cut_duration']}s)")


# ---------------------------------------------------------------- subtitles

DEFAULT_PROFILE = HERE / "netflix_ko.json"
_SENT_END = re.compile(r"[.?!…]$")


def clean_token(tok: str, prof: dict) -> str:
    if prof.get("commas_to_space"):
        tok = tok.replace(",", "")
    return tok


def split_lines(text_words: list[str], max_chars: int, max_lines: int):
    """어절 단위로 줄바꿈. 줄 길이 차이를 최소화. 불가능하면 None."""
    full = " ".join(text_words)
    if len(full) <= max_chars:
        return [full]
    if max_lines < 2:
        return None
    best = None
    for k in range(1, len(text_words)):
        l1, l2 = " ".join(text_words[:k]), " ".join(text_words[k:])
        if len(l1) <= max_chars and len(l2) <= max_chars:
            score = abs(len(l1) - len(l2))
            if best is None or score < best[0]:
                best = (score, [l1, l2])
    return best[1] if best else None


def make_cues(words: list[dict], prof: dict) -> list[dict]:
    mc, ml = prof["max_chars_per_line"], prof["max_lines"]
    cues, cur = [], []

    def flush():
        nonlocal cur
        if cur:
            cues.append(cur)
            cur = []

    for w in words:
        tok = clean_token(w["text"], prof)
        if not tok:
            continue
        cand = cur + [{**w, "text": tok}]
        too_long = split_lines([x["text"] for x in cand], mc, ml) is None
        too_slow_gap = bool(cur) and w["start"] - cur[-1]["end"] > prof["max_word_gap"]
        too_dur = bool(cur) and (w["end"] - cur[0]["start"]) > prof["max_duration"]
        jump = bool(cur) and w.get("seg") != cur[-1].get("seg")  # 점프컷 경계에서는 자막을 끊는다
        if cur and (too_long or too_slow_gap or too_dur or jump):
            flush()
            cand = [{**w, "text": tok}]
        cur = cand
        if _SENT_END.search(w["text"]):
            flush()
    flush()

    out = []
    for c in cues:
        lines = split_lines([x["text"] for x in c], mc, ml)
        if lines is None:  # 한 어절이 한 줄 상한을 넘는 경우: 그대로 두고 경고
            lines = [" ".join(x["text"] for x in c)]
        if prof.get("strip_trailing_period"):
            lines[-1] = lines[-1].rstrip(".")
        out.append({"start": c[0]["start"], "end": c[-1]["end"], "lines": lines})
    return out


def apply_timing(cues: list[dict], prof: dict, total: float):
    fps = prof["fps"]
    gap = prof["min_gap_frames"] / fps
    warns = []
    for i, c in enumerate(cues):
        nxt = cues[i + 1]["start"] if i + 1 < len(cues) else total
        end = c["end"] + prof["tail_pad"]
        end = max(end, c["start"] + prof["min_duration"])
        end = min(end, nxt - gap if i + 1 < len(cues) else total)
        c["end"] = max(end, c["start"] + 0.2)
        dur = c["end"] - c["start"]
        chars = sum(len(l) for l in c["lines"])
        c["cps"] = round(chars / dur, 1)
        if dur < prof["min_duration"] - 1e-6:
            warns.append((i + 1, f"min_duration 미달 {dur:.2f}s"))
        if c["cps"] > prof["max_cps"]:
            warns.append((i + 1, f"읽기속도 {c['cps']} cps > {prof['max_cps']}"))
        if any(len(l) > prof["max_chars_per_line"] for l in c["lines"]):
            warns.append((i + 1, "줄 길이 초과(긴 어절)"))
    return cues, warns


def fmt_ts(t: float) -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def to_srt(cues: list[dict]) -> str:
    rows = []
    for i, c in enumerate(cues, 1):
        rows.append(f"{i}\n{fmt_ts(c['start'])} --> {fmt_ts(c['end'])}\n" + "\n".join(c["lines"]) + "\n")
    return "\n".join(rows)


def cmd_subtitles(a) -> None:
    work = Path(a.work)
    prof = load_json(Path(a.profile))
    tr, plan = load_json(work / "transcript.json"), load_json(work / "plan.json")
    remap = remap_factory(plan["keep"])
    words = []
    for w in tr["words"]:
        mid = (w["start"] + w["end"]) / 2
        m, seg = remap(mid, True)
        if m is None:
            continue  # 잘려나간 단어
        s, e = remap(w["start"]), remap(w["end"])
        s = s if s is not None else m
        e = e if e is not None else s + 0.1
        words.append({"text": w["text"], "start": s, "end": max(e, s + 0.05), "seg": seg})
    cues = make_cues(words, prof)
    cues, warns = apply_timing(cues, prof, plan["cut_duration"])
    save_json(work / "cues.json", cues)
    (work / "cut.srt").write_text(to_srt(cues), encoding="utf-8")
    print(f"[subtitles] {len(cues)} cues -> {work/'cut.srt'}  (프로필: {prof['_name']})")
    for n, msg in warns:
        print(f"  ! cue {n}: {msg}")
    if not warns:
        print("  규격 위반 없음")


# ---------------------------------------------------------------- compose

SFX_SPECS = {
    # name: ffmpeg lavfi 합성 (earshot 등 외부 효과음이 없을 때 기본값)
    "whoosh": ["-f", "lavfi", "-i", "anoisesrc=d=0.35:c=pink:r=44100",
               "-af", "highpass=f=800,lowpass=f=6000,afade=t=in:d=0.12,afade=t=out:st=0.15:d=0.2,volume=0.8"],
    "pop": ["-f", "lavfi", "-i", "sine=f=880:d=0.12:r=44100",
            "-af", "afade=t=out:st=0.01:d=0.11,volume=0.7"],
}


def ensure_sfx(sfx_dir: Path | None, dest: Path) -> dict[str, str]:
    """카테고리(whoosh/pop)별 효과음 파일을 프로젝트 sfx/ 로 준비."""
    dest.mkdir(parents=True, exist_ok=True)
    chosen: dict[str, str] = {}
    for name, spec in SFX_SPECS.items():
        src = None
        if sfx_dir and sfx_dir.is_dir():
            cands = sorted(p for p in sfx_dir.iterdir()
                           if p.suffix.lower() in {".wav", ".mp3", ".m4a", ".ogg"}
                           and p.stem.lower().startswith(name))
            src = cands[0] if cands else None
        target = dest / f"{name}.wav"
        if src:
            run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-ar", "44100", str(target)])
        else:
            run(["ffmpeg", "-y", "-loglevel", "error", *spec, str(target)])
        chosen[name] = f"sfx/{name}.wav"
    return chosen


def plan_sfx(plan_keep, cues, *, min_spacing: float, max_whoosh: int):
    """효과음 배치: 점프컷 지점=whoosh(간격 제한), 숫자가 든 자막 시작=pop."""
    events, last = [], -1e9
    acc = 0.0
    for s, e in plan_keep[:-1]:
        acc += e - s
        if acc - last >= min_spacing and sum(1 for x in events if x["kind"] == "whoosh") < max_whoosh:
            events.append({"kind": "whoosh", "t": round(acc, 3)})
            last = acc
    for c in cues:
        if re.search(r"\d", " ".join(c["lines"])) and all(abs(c["start"] - x["t"]) > 0.3 for x in events):
            events.append({"kind": "pop", "t": round(c["start"], 3)})
    return sorted(events, key=lambda x: x["t"])


COMPOSE_TMPL = """<!doctype html>
<html lang="ko">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=__W__, height=__H__" />
    <script src="__GSAP__"></script>
    <style>
      * { margin: 0; padding: 0; box-sizing: border-box; }
      html, body { width: __W__px; height: __H__px; overflow: hidden; background: #000; }
__FONTFACE__
      body { font-family: "Pretendard", "Noto Sans KR", sans-serif; }
      .cap {
        position: absolute; left: 0; right: 0; bottom: __BOTTOM__px;
        text-align: center; opacity: 0;
      }
      .cap span {
        display: inline-block; padding: 10px 28px; border-radius: 12px;
        background: rgba(0, 0, 0, 0.62); color: #fff; font-weight: 700;
        font-size: __FS__px; line-height: 1.35; white-space: pre-line;
        text-shadow: 0 2px 6px rgba(0, 0, 0, 0.5);
      }
    </style>
  </head>
  <body>
    <div id="root" data-composition-id="main" data-start="0" data-duration="__DUR__"
         data-width="__W__" data-height="__H__">
      <video id="a-roll" class="clip" src="cut.mp4" muted playsinline
             data-start="0" data-duration="__DUR__" data-track-index="0"
             style="position:absolute;inset:0;width:100%;height:100%;object-fit:cover"></video>
      <audio id="a-roll-audio" src="cut.mp4" data-start="0" data-duration="__DUR__"
             data-track-index="2" data-volume="1"></audio>
__SFX__
__CAPS__
      <script>
        window.__timelines = window.__timelines || {};
        const tl = gsap.timeline({ paused: true });
__TWEENS__
        window.__timelines["main"] = tl;
      </script>
    </div>
  </body>
</html>
"""


def cmd_compose(a) -> None:
    work = Path(a.work)
    proj = Path(a.project)
    plan, cues = load_json(work / "plan.json"), load_json(work / "cues.json")
    cut = work / "cut.mp4"
    proj.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cut, proj / "cut.mp4")
    shutil.copy2(work / "cut.srt", proj / "cut.srt")

    dur = probe_duration(cut)
    sfx_files = ensure_sfx(Path(a.sfx_dir) if a.sfx_dir else None, proj / "sfx") if not a.no_sfx else {}
    events = [] if a.no_sfx else plan_sfx(plan["keep"], cues, min_spacing=a.sfx_spacing, max_whoosh=a.max_whoosh)

    sfx_html = []
    for i, ev in enumerate(events):
        sfx_html.append(
            f'      <audio id="sfx{i}" src="{sfx_files[ev["kind"]]}" data-start="{ev["t"]}" '
            f'data-duration="0.5" data-track-index="{3 + (i % 2)}" data-volume="{a.sfx_volume}"></audio>'
        )
    cap_html = []
    for i, c in enumerate(cues):
        txt = html.escape("\n".join(c["lines"]))
        cap_html.append(
            f'      <div id="cap{i}" class="cap clip" data-start="{c["start"]:.3f}" '
            f'data-duration="{c["end"] - c["start"]:.3f}" data-track-index="1"><span>{txt}</span></div>'
        )
    tweens = []
    for i, c in enumerate(cues):
        d = c["end"] - c["start"]
        fade_in, fade_out = min(0.12, d / 3), min(0.08, d / 4)
        t_out = c["start"] + d - fade_out
        tweens.append(
            f'        tl.fromTo("#cap{i}", {{ opacity: 0, y: 12 }}, '
            f'{{ opacity: 1, y: 0, duration: {fade_in:.3f}, ease: "power2.out" }}, {c["start"]:.3f});\n'
            f'        tl.to("#cap{i}", {{ opacity: 0, duration: {fade_out:.3f}, ease: "power1.in" }}, {t_out:.3f});\n'
            f'        tl.set("#cap{i}", {{ opacity: 0 }}, {c["end"]:.3f});'
        )
    fontface = ""
    if a.font_file:
        ff = Path(a.font_file)
        (proj / "fonts").mkdir(exist_ok=True)
        shutil.copy2(ff, proj / "fonts" / ff.name)
        fontface = ('      @font-face { font-family: "Pretendard"; font-weight: 700; '
                    f'src: url("fonts/{ff.name}"); }}')
    gsap_src = "https://cdn.jsdelivr.net/npm/gsap@3.14.2/dist/gsap.min.js"
    if a.gsap_file:  # CDN 이 막힌 환경(폐쇄망/샌드박스)용
        shutil.copy2(a.gsap_file, proj / "gsap.min.js")
        gsap_src = "gsap.min.js"
    page = (COMPOSE_TMPL
            .replace("__GSAP__", gsap_src)
            .replace("__TWEENS__", "\n".join(tweens))
            .replace("__FONTFACE__", fontface)
            .replace("__W__", str(a.width)).replace("__H__", str(a.height))
            .replace("__DUR__", f"{dur:.3f}")
            .replace("__BOTTOM__", str(int(a.height * 0.09)))
            .replace("__FS__", str(int(a.height * 0.047)))
            .replace("__SFX__", "\n".join(sfx_html))
            .replace("__CAPS__", "\n".join(cap_html))
            )
    (proj / "index.html").write_text(page, encoding="utf-8")
    print(f"[compose] {proj/'index.html'}  (효과음 {len(events)}개, 자막 {len(cues)}개)")
    print(f"  다음: npx hyperframes lint {proj} && npx hyperframes render {proj} --output final.mp4")


# ---------------------------------------------------------------- CLI


def add_common(p):
    p.add_argument("video", help="원본 영상")
    p.add_argument("--work", default="work", help="중간 산출물 폴더")


def add_transcribe(p):
    p.add_argument("--model", default="large-v3-turbo")
    p.add_argument("--language", default="ko")
    p.add_argument("--device", default="auto")
    p.add_argument("--compute-type", default="auto", dest="compute_type",
                   help="저사양 CPU 는 int8 권장")


def add_plan(p):
    p.add_argument("--noise-db", type=float, default=-35.0, dest="noise_db")
    p.add_argument("--min-silence", type=float, default=0.6, dest="min_silence")
    p.add_argument("--keep-pad", type=float, default=0.12, dest="keep_pad",
                   help="무음 컷 후 양끝에 남기는 호흡(초)")
    p.add_argument("--min-keep", type=float, default=0.25, dest="min_keep")
    p.add_argument("--retake-gap", type=float, default=0.5, dest="retake_gap",
                   help="이 이상 쉬면 새 발화로 간주")
    p.add_argument("--retake-thr", type=float, default=0.6, dest="retake_thr")
    p.add_argument("--lookahead", type=int, default=2)
    p.add_argument("--min-chars", type=int, default=4, dest="min_chars")
    p.add_argument("--pre-roll", type=float, default=0.05, dest="pre_roll")


def add_sub(p):
    p.add_argument("--profile", default=str(DEFAULT_PROFILE))


def add_compose(p):
    p.add_argument("--project", default="project")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--sfx-dir", dest="sfx_dir", default=None,
                   help="whoosh*/pop* 로 시작하는 효과음 폴더 (예: earshot 산출물)")
    p.add_argument("--font-file", dest="font_file", default=None,
                   help="Pretendard Bold woff2/ttf 경로 (없으면 렌더 시 기본 폰트로 대체됨)")
    p.add_argument("--gsap-file", dest="gsap_file", default=None,
                   help="로컬 gsap.min.js (CDN 접속이 막힌 환경에서 필수)")
    p.add_argument("--no-sfx", action="store_true", dest="no_sfx")
    p.add_argument("--sfx-volume", type=float, default=0.35, dest="sfx_volume")
    p.add_argument("--sfx-spacing", type=float, default=4.0, dest="sfx_spacing")
    p.add_argument("--max-whoosh", type=int, default=40, dest="max_whoosh")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name, fn, adders in [
        ("transcribe", cmd_transcribe, [add_common, add_transcribe]),
        ("plan", cmd_plan, [add_common, add_plan]),
        ("cut", cmd_cut, [add_common]),
        ("subtitles", cmd_subtitles, [add_common, add_sub]),
        ("compose", cmd_compose, [add_common, add_compose]),
        ("all", None, [add_common, add_transcribe, add_plan, add_sub, add_compose]),
    ]:
        p = sp.add_parser(name)
        for ad in adders:
            ad(p)
        p.set_defaults(fn=fn)
    a = ap.parse_args(argv)
    if a.cmd == "all":
        for fn in (cmd_transcribe, cmd_plan, cmd_cut, cmd_subtitles, cmd_compose):
            fn(a)
    else:
        a.fn(a)


if __name__ == "__main__":
    main()
