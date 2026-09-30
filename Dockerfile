FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /workspace

RUN apt-get update && apt-get install -y ffmpeg git git-lfs && rm -rf /var/lib/apt/lists/*
ENV FFMPEG_PATH=/usr/bin

RUN git clone https://github.com/antgroup/echomimic_v2.git /workspace/echomimic_v2
WORKDIR /workspace/echomimic_v2

RUN sed -i 's/onnxruntime-gpu==1.20.1/onnxruntime-gpu==1.16.3/' requirements.txt && \
    pip install --no-cache-dir -r requirements.txt

# Веса модели с HuggingFace (BadToBest/EchoMimicV2). ВАЖНО: конфиг
# configs/prompts/infer.yaml, зашитый в репозиторий, ссылается на
# конкретные относительные пути к весам (pretrained_vae_path,
# pretrained_base_model_path, motion_module_path, pose_encoder_path,
# audio_model_path) — эта команда скачивает веса в pretrained_weights/
# как в оригинальном README проекта. Если после сборки инференс падает
# с "file not found" на одном из этих путей — открой
# configs/prompts/infer.yaml внутри контейнера и сверь пути с реальной
# структурой папок, которую создал snapshot_download.
RUN pip install --no-cache-dir "huggingface_hub[cli]" && \
    python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='BadToBest/EchoMimicV2', local_dir='pretrained_weights')"

# Патч: зацикливание pose-последовательности + снятие лишнего
# ограничения по длине клипа числом файлов в pose-папке (см. сам
# скрипт патча — без него клип обрывается по длине САМОЙ КОРОТКОЙ
# pose-последовательности, а не по длине аудио/текста)
COPY patch_echomimic_pose_loop.py /workspace/echomimic_v2/patch_echomimic_pose_loop.py
RUN python /workspace/echomimic_v2/patch_echomimic_pose_loop.py

# transformers/librosa/soundfile — для автоопределения эмоции по аудио
# в Кейсе 2 Pro (см. handler.py, detect_emotion_from_audio)
RUN pip install --no-cache-dir transformers librosa soundfile runpod

# Прогреваем кэш модели распознавания эмоций на этапе сборки образа —
# чтобы холодный старт воркера не тратил время на скачивание весов
# этой модели при первом реальном запросе
RUN python -c "from transformers import pipeline; pipeline('audio-classification', model='ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition')"

RUN mkdir -p /workspace/tmp

COPY handler.py /workspace/echomimic_v2/handler.py

CMD ["python", "-u", "handler.py"]
