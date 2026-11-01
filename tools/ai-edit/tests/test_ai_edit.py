import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ai_edit as ae  # noqa: E402


def W(text, s, e):
    return {"text": text, "start": s, "end": e}


class RetakeTests(unittest.TestCase):
    def test_partial_restart_is_dropped_keep_later(self):
        words = [W("오늘은", 0.2, 0.6), W("AI", 0.7, 1.0), W("편집에", 1.1, 1.8),
                 W("오늘은", 4.2, 4.6), W("AI", 4.7, 5.0), W("영상", 5.1, 5.4),
                 W("편집에", 5.5, 5.8), W("대해", 5.9, 6.1)]
        utts = ae.group_utterances(words, 0.5)
        self.assertEqual(len(utts), 2)
        drops = ae.find_retakes(utts, 0.6, 2, 4)
        self.assertEqual([d["index"] for d in drops], [0])

    def test_distinct_sentences_not_dropped(self):
        words = [W("첫째", 0, 0.4), W("무음을", 0.5, 0.9), W("자릅니다", 1.0, 1.5),
                 W("둘째", 3.0, 3.4), W("자막을", 3.5, 3.9), W("입힙니다", 4.0, 4.5)]
        drops = ae.find_retakes(ae.group_utterances(words, 0.5), 0.6, 2, 4)
        self.assertEqual(drops, [])

    def test_no_pause_means_same_utterance(self):
        words = [W("안녕", 0, 0.3), W("안녕", 0.35, 0.7)]
        self.assertEqual(len(ae.group_utterances(words, 0.5)), 1)


class IntervalTests(unittest.TestCase):
    def test_invert_and_merge(self):
        cuts = ae.merge_intervals([(2, 3), (2.5, 4), (6, 7)])
        self.assertEqual(cuts, [(2, 4), (6, 7)])
        self.assertEqual(ae.invert(cuts, 9), [(0.0, 2), (4, 6), (7, 9)])

    def test_remap(self):
        r = ae.remap_factory([[0, 2], [4, 6]])
        self.assertAlmostEqual(r(1), 1)
        self.assertAlmostEqual(r(5), 3)
        self.assertIsNone(r(3))


class SubtitleTests(unittest.TestCase):
    prof = json.loads((Path(__file__).resolve().parents[1] / "netflix_ko.json").read_text())

    def test_lines_within_limit(self):
        words = [W(t, i * 0.4, i * 0.4 + 0.35) for i, t in enumerate(
            "오늘은 클로드 코드로 영상 편집을 자동화하는 방법을 알려드리겠습니다".split())]
        cues = ae.make_cues(words, self.prof)
        cues, _ = ae.apply_timing(cues, self.prof, 10)
        for c in cues:
            self.assertLessEqual(len(c["lines"]), 2)
            for line in c["lines"]:
                self.assertLessEqual(len(line), 16)

    def test_min_duration_and_gap(self):
        words = [W("네", 0, 0.2), W("좋아요", 1.0, 1.3)]
        cues, _ = ae.apply_timing(ae.make_cues(words, self.prof), self.prof, 5)
        self.assertGreaterEqual(cues[0]["end"] - cues[0]["start"], 0.8)
        self.assertLessEqual(cues[0]["end"], cues[1]["start"] - 2 / 30 + 1e-6)

    def test_period_stripped(self):
        cues = ae.make_cues([W("끝입니다.", 0, 1)], self.prof)
        self.assertEqual(cues[0]["lines"], ["끝입니다"])


class EndToEnd(unittest.TestCase):
    """합성 영상(톤/무음 구간) + 가짜 transcript 로 plan->cut->subtitles->compose."""

    def test_pipeline(self):
        tmp = Path(tempfile.mkdtemp())
        src = tmp / "src.mp4"
        # 0-2 톤, 2-4 무음, 4-6 톤, 6-7 무음, 7-9 톤
        expr = "between(t,0,2)+between(t,4,6)+between(t,7,9)"
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "color=c=gray:s=640x360:r=30:d=9",
            "-f", "lavfi", "-i", f"aevalsrc='0.5*sin(2*PI*440*t)*({expr})':s=44100:d=9",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(src)], check=True)
        work = tmp / "work"
        work.mkdir()
        words = [W("오늘은", 0.2, 0.6), W("AI", 0.7, 1.0), W("편집에", 1.1, 1.8),
                 W("오늘은", 4.2, 4.6), W("AI", 4.7, 5.0), W("영상", 5.1, 5.4),
                 W("편집에", 5.5, 5.8), W("대해", 5.85, 5.95),
                 W("조회수", 7.2, 7.6), W("190K", 7.7, 8.2), W("입니다.", 8.3, 8.8)]
        (work / "transcript.json").write_text(json.dumps(
            {"language": "ko", "duration": 9.0, "words": words}, ensure_ascii=False))

        ae.main(["plan", str(src), "--work", str(work)])
        plan = json.loads((work / "plan.json").read_text())
        self.assertEqual(len(plan["retakes_dropped"]), 1)
        self.assertLess(plan["cut_duration"], 5.5)
        self.assertGreater(plan["cut_duration"], 3.5)

        ae.main(["cut", str(src), "--work", str(work)])
        got = ae.probe_duration(work / "cut.mp4")
        self.assertAlmostEqual(got, plan["cut_duration"], delta=0.15)

        ae.main(["subtitles", str(src), "--work", str(work)])
        srt = (work / "cut.srt").read_text(encoding="utf-8")
        self.assertIn("190K", srt)
        # 삭제된 첫 발화(0.2~1.8)의 자막은 없어야 함: 'AI 편집에' 로 끝나는 큐가 1개만
        self.assertEqual(srt.count("대해"), 1)

        ae.main(["compose", str(src), "--work", str(work), "--project", str(tmp / "proj")])
        page = (tmp / "proj" / "index.html").read_text(encoding="utf-8")
        self.assertIn('data-composition-id="main"', page)
        self.assertIn("window.__timelines", page)
        self.assertTrue((tmp / "proj" / "sfx" / "pop.wav").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
