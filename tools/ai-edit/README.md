# ai-edit — 영상 편집 자동화 파이프라인 (HyperFrames)

촬영 → 재테이크/무음 컷 → 자막 → 효과음 → 렌더를 한 번에.

## 설치

```bash
sudo apt-get install -y ffmpeg          # macOS: brew install ffmpeg
pip install faster-whisper              # large-v3-turbo (CPU면 --compute-type int8)
bun install && bun run build            # hyperframes 레포 루트에서 (렌더용)
```

## 사용

```bash
cd tools/ai-edit
python3 ai_edit.py all my_take.mp4 --work work --project project --compute-type int8
# 또는 단계별: transcribe -> plan -> cut -> subtitles -> compose
npx hyperframes lint project
npx hyperframes render project --output final.mp4
```

산출물: `work/transcript.json`, `work/plan.json`(삭제 목록 포함), `work/cut.mp4`, `work/cut.srt`, `project/index.html`.

## 촬영 규칙 (재테이크 탐지의 전제)

말 실수 → **0.5초 이상 쉬고** 해당 문장을 처음부터 다시. `--retake-gap`(기본 0.5s) 이상 쉰 뒤 앞 발화와
유사도 `--retake-thr`(기본 0.6)↑ 인 발화가 오면 **앞 발화를 삭제**하고 뒤를 살린다.

## 반드시 확인할 것

1. `plan` 출력의 "삭제 목록"을 눈으로 확인 (오탐 시 `--retake-thr` 상향). 의도된 반복("좋아요 좋아요")은 오탐 가능.
2. `netflix_ko.json` 은 **원문 미확인 값**(16자/줄, 2줄, 12 cps, 최소 0.833s 등). 넷플릭스 한국어 Timed Text Style Guide 와 대조해 수정.
3. 효과음: `--sfx-dir` 에 `whoosh*.wav`, `pop*.wav` 를 두면 사용. 없으면 ffmpeg 합성음으로 대체.
   Leonxlnx/earshot 은 미검증 — 산출물을 위 이름으로 맞춰 넣으면 연결됨.
4. 폰트: Pretendard Bold 를 `--font-file` 로 지정(미지정 시 lint 경고 + 대체 폰트).
5. CDN 이 막힌 환경에선 `--gsap-file path/to/gsap.min.js` 필수.

## 테스트

```bash
python3 -m unittest discover -s tests -v
```
