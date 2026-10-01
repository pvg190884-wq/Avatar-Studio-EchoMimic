FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /workspace

RUN apt-get update && apt-get install -y ffmpeg git git-lfs wget && rm -rf /var/lib/apt/lists/*
ENV FFMPEG_PATH=/usr/bin

RUN git clone https://github.com/antgroup/echomimic_v2.git /workspace/echomimic_v2
WORKDIR /workspace/echomimic_v2

# Удаляем системный blinker, чтобы pip мог поставить свою версию,
# и фиксируем onnxruntime-gpu на совместимой версии 1.16.3
RUN apt-get remove -y python3-blinker || true && \
    sed -i 's/onnxruntime-gpu==1.20.1/onnxruntime-gpu==1.16.3/' requirements.txt && \
    pip install --no-cache-dir -r requirements.txt

# Веса модели. ВАЖНО (обнаружено на реальном тесте): BadToBest/EchoMimicV2
# сам по себе НЕ содержит рабочих sd-vae-ft-mse и
# sd-image-variations-diffusers — в нём это пустые папки-заглушки.
# Официальный README требует докачать эти две модели отдельно с их
# собственных HuggingFace-репозиториев поверх основного скачивания, plus
# Whisper tiny.pt для audio_processor (см. linux_setup.sh апстрима).
# Без этого шага infer.py падает на самом первом обращении к VAE:
# "OSError: ...sd-vae-ft-mse is not the path to a directory containing
# a config.json file".
RUN pip install --no-cache-dir "huggingface_hub[cli]" && \
    python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='BadToBest/EchoMimicV2', local_dir='pretrained_weights')" && \
    python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='stabilityai/sd-vae-ft-mse', local_dir='pretrained_weights/sd-vae-ft-mse')" && \
    python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='lambdalabs/sd-image-variations-diffusers', local_dir='pretrained_weights/sd-image-variations-diffusers')" && \
    mkdir -p pretrained_weights/audio_processor && \
    wget -O pretrained_weights/audio_processor/tiny.pt \
      https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt

# Патч: зацикливание pose-последовательности + снятие лишнего
# ограничения по длине клипа числом файлов в pose-папке (см. сам
# скрипт патча — без него клип обрывается по длине САМОЙ КОРОТКОЙ
# pose-последовательности, а не по длине аудио/текста)
COPY patch_echomimic_pose_loop.py /workspace/echomimic_v2/patch_echomimic_pose_loop.py
RUN python /workspace/echomimic_v2/patch_echomimic_pose_loop.py

# transformers/librosa/soundfile — для автоопределения эмоции по аудио
# в Кейсе 2 Pro (см. handler.py, detect_emotion_from_audio)
RUN pip install --no-cache-dir transformers librosa soundfile runpod Pillow

# Прогреваем кэш модели распознавания эмоций на этапе сборки образа —
# чтобы холодный старт воркера не тратил время на скачивание весов
# этой модели при первом реальном запросе
RUN python -c "from transformers import pipeline; pipeline('audio-classification', model='ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition')"

RUN mkdir -p /workspace/tmp

COPY handler.py /workspace/echomimic_v2/handler.py

CMD ["python", "-u", "handler.py"]
