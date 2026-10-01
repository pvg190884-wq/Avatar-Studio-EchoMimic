import os
import glob
import time
import base64
import shutil
import subprocess
import tempfile
import uuid

from PIL import Image, ImageOps
import runpod

REPO_DIR = "/workspace/echomimic_v2"
OUTPUTS_DIR = os.path.join(REPO_DIR, "outputs")
TMP_ROOT = "/workspace/tmp"

# Маппинг эмоции — явной от пользователя (Кейс 1 Pro) или определённой
# автоматически по аудио (Кейс 2 Pro) — на готовые pose-последовательности
# из комплекта EchoMimicV2 (assets/halfbody_demo/pose/*). Метки calm/
# disgust/fearful не встречаются в явном UI Кейса 1, но их отдаёт модель
# распознавания эмоций в Кейсе 2, поэтому замаплены и они.
EMOTION_TO_POSE = {
    "neutral": "01",
    "calm": "01",
    "happy": "good",
    "surprised": "ultraman",
    "angry": "fight",
    "disgust": "fight",
    "sad": "02",
    "fearful": "02",
}
DEFAULT_POSE = "01"

_emotion_classifier = None


def get_emotion_classifier():
    """Ленивая инициализация — модель грузится один раз на холодном
    старте воркера (веса уже прогреты в образ на этапе сборки, см.
    Dockerfile) и переиспользуется между запросами, пока RunPod держит
    контейнер тёплым (idleTimeout)."""
    global _emotion_classifier
    if _emotion_classifier is None:
        from transformers import pipeline
        _emotion_classifier = pipeline(
            "audio-classification",
            model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
        )
    return _emotion_classifier


def detect_emotion_from_audio(audio_path: str) -> str:
    """Кейс 2 Pro: эмоция не передаётся явно, определяем её по тону
    голоса. Модель обучена на английской речи (датасет RAVDESS), эмоция
    определяется в первую очередь по просодии (интонация, темп, высота
    тона), а не по смыслу слов — на русской речи точность ниже, чем на
    английской, но для выбора жеста (а не точной эмоциональной оценки)
    этого достаточно."""
    classifier = get_emotion_classifier()
    results = classifier(audio_path)
    return results[0]["label"].lower()


def resolve_pose_name(emotion: str | None, audio_path: str) -> str:
    if emotion:
        return EMOTION_TO_POSE.get(emotion.lower(), DEFAULT_POSE)
    detected = detect_emotion_from_audio(audio_path)
    return EMOTION_TO_POSE.get(detected, DEFAULT_POSE)


def prepare_square_image(image_path: str, size: int = 768) -> None:
    """ВТОРАЯ НАХОДКА НА РЕАЛЬНОМ ТЕСТЕ: нельзя просто передать
    непрямоугольные -W/-H в infer.py — внутри пайплайна есть
    pose-маска для анимации рук (tgt_musk), геометрически зашитая под
    КВАДРАТНЫЙ холст (сами .npy pose-последовательности рассчитаны под
    768x768) — непрямоугольный холст ломает её с ValueError при
    наложении маски.

    Значит холст обязан остаться квадратным, но растягивать
    прямоугольное фото под квадрат (как делает infer.py по умолчанию)
    — и есть первопричина сильных искажений с первого теста. Решение:
    вписываем фото в квадратный холст с полями (letterbox), сохраняя
    пропорции и НЕ обрезая края — в отличие от обрезки по центру, это
    гарантированно не отрежет руки, которые как раз должна анимировать
    Pro-версия."""
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        squared = ImageOps.pad(img, (size, size), method=Image.LANCZOS,
                                color=(0, 0, 0), centering=(0.5, 0.5))
        squared.save(image_path)


def run_echomimic_inference(image_path: str, audio_path: str, pose_name: str,
                             max_seconds: float, fps: int = 24) -> str:
    """Запускает оригинальный infer.py EchoMimicV2 как подпроцесс.
    -L задаём с запасом сверх max_seconds — реальная длина всё равно
    обрежется по фактической длительности аудио (args.L = min(...)
    внутри infer.py); патч patch_echomimic_pose_loop.py убирает
    лишнее ограничение по числу файлов в pose-папке, так что запас
    здесь не даёт клипу оборваться раньше времени из-за короткой
    pose-последовательности."""
    os.makedirs(TMP_ROOT, exist_ok=True)
    job_dir = tempfile.mkdtemp(prefix="echomimic_job_", dir=TMP_ROOT)
    ref_images_dir = os.path.join(job_dir, "ref")
    audio_dir = os.path.join(job_dir, "audio")
    os.makedirs(ref_images_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)

    # ВАЖНО (обнаружено на реальном тесте): refimg_name/audio_name должны
    # содержать хотя бы один уровень подпапки — infer.py строит ref_flag
    # через split('/')[-2], и при плоском имени файла без подпапки падает
    # с IndexError: list index out of range.
    refimg_name = "sample/ref.png"
    audio_name = "sample/audio.wav"
    os.makedirs(os.path.join(ref_images_dir, "sample"), exist_ok=True)
    os.makedirs(os.path.join(audio_dir, "sample"), exist_ok=True)
    shutil.copy(image_path, os.path.join(ref_images_dir, refimg_name))
    shutil.copy(audio_path, os.path.join(audio_dir, audio_name))
    prepare_square_image(os.path.join(ref_images_dir, refimg_name))

    frame_limit = max(1, int(max_seconds * fps) + fps)  # +1 сек запаса

    start_time = time.time()
    cmd = [
        "python", "infer.py",
        "--config", "./configs/prompts/infer.yaml",
        "--ref_images_dir", ref_images_dir,
        "--audio_dir", audio_dir,
        "--refimg_name", refimg_name,
        "--audio_name", audio_name,
        "--pose_name", pose_name,
        "-L", str(frame_limit),
        "--fps", str(fps),
    ]
    result = subprocess.run(cmd, cwd=REPO_DIR, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"EchoMimicV2 inference упал: {result.stderr[-3000:]}")

    # infer.py сохраняет результат по сложному пути, зависящему от
    # motion_module_path/seed/имён файлов — вместо того чтобы
    # воспроизводить эту логику здесь (и рисковать разъехаться при
    # обновлении апстрима), просто берём самый свежий *_sig.mp4,
    # появившийся после старта этого подпроцесса
    candidates = glob.glob(os.path.join(OUTPUTS_DIR, "**", "*_sig.mp4"), recursive=True)
    candidates = [c for c in candidates if os.path.getmtime(c) >= start_time]
    if not candidates:
        raise RuntimeError("EchoMimicV2 не создал видео — файл *_sig.mp4 не найден после инференса")
    output_path = max(candidates, key=os.path.getmtime)

    shutil.rmtree(job_dir, ignore_errors=True)
    return output_path


def handler(event):
    inp = event.get("input", {})
    image_b64 = inp.get("image_base64")
    audio_b64 = inp.get("audio_base64")
    emotion = inp.get("emotion")  # None => автоопределение (Кейс 2 Pro)
    max_seconds = float(inp.get("max_seconds", 15))

    if not image_b64 or not audio_b64:
        return {"error": "image_base64 и audio_base64 обязательны"}

    os.makedirs(TMP_ROOT, exist_ok=True)
    request_id = uuid.uuid4().hex
    image_path = f"{TMP_ROOT}/{request_id}_ref.png"
    audio_path = f"{TMP_ROOT}/{request_id}_audio.wav"

    with open(image_path, "wb") as f:
        f.write(base64.b64decode(image_b64))
    with open(audio_path, "wb") as f:
        f.write(base64.b64decode(audio_b64))

    try:
        pose_name = resolve_pose_name(emotion, audio_path)
        output_path = run_echomimic_inference(image_path, audio_path, pose_name, max_seconds)
        with open(output_path, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode("utf-8")
        return {"video_base64": video_b64, "pose_used": pose_name}
    except Exception as e:
        return {"error": str(e)}
    finally:
        for p in (image_path, audio_path):
            if os.path.exists(p):
                os.remove(p)


runpod.serverless.start({"handler": handler})
