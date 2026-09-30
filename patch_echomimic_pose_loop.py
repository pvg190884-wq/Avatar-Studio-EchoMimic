"""Патч для infer.py EchoMimicV2 — применяется на этапе сборки Docker-
образа (тот же паттерн build-time-патчей, что уже используется в
Avatar-Studio-Lipsync для MuseTalk: patch_musetalk_audio.py,
patch_musetalk_bbox.py — поиск по содержимому строки, а не по точному
совпадению отступов, т.к. форматирование исходника со временем может
измениться).

БЕЗ этого патча в оригинальном infer.py:

1) итоговая длина видео жёстко ограничена количеством .npy-файлов в
   выбранной pose-папке:
       args.L = min(args.L, длительность_аудио*fps, число_файлов_в_pose)
   Если pose-последовательность короче, чем нужно для полной
   длительности аудио/текста (а у некоторых готовых pose-папок в
   комплекте EchoMimicV2 всего ~240 кадров = 10 сек при 24 fps), ролик
   молча обрежется до длины pose-последовательности — раньше, чем
   закончится речь. Для продукта, где заявлен клип до 15 секунд, это
   реальная проблема, а не косметика.

2) кадр pose берётся строго по индексу index.npy без каких-либо
   проверок — при выходе за пределы папки упадёт FileNotFoundError
   (что на практике не должно было случаться из-за пункта 1, но всё
   равно хрупко).

Патч убирает ограничение по числу pose-файлов из формулы args.L и
зацикливает выбор pose-кадра по модулю числа доступных файлов — жест
просто повторяется по кругу, пока не закончится реальная длительность
аудио. Так Pro-режим честно отрабатывает до ~15 секунд (лимит продукта)
независимо от длины конкретной pose-последовательности."""

INFER_PY_PATH = "/workspace/echomimic_v2/infer.py"

with open(INFER_PY_PATH, "r", encoding="utf-8") as f:
    content = f.read()

# 1) убираем ограничение по числу файлов в pose-папке из формулы args.L
old_l_line = "args.L = min(args.L, int(audio_clip.duration * final_fps), len(os.listdir(inputs_dict['pose'])))"
new_l_line = (
    "args.L = min(args.L, int(audio_clip.duration * final_fps))  "
    "# patched by patch_echomimic_pose_loop.py: не ограничиваем длину числом pose-файлов"
)
if old_l_line not in content:
    raise RuntimeError(
        "patch_echomimic_pose_loop.py: не найдена ожидаемая строка 'args.L = min(...)' — "
        "апстрим infer.py изменился, нужно сверить патч вручную"
    )
content = content.replace(old_l_line, new_l_line)

# 2) зацикливаем выбор pose-кадра по модулю числа файлов в папке
old_pose_line = "tgt_musk_path = os.path.join(inputs_dict['pose'], \"{}.npy\".format(index))"
new_pose_block = (
    "pose_file_count = len([fn for fn in os.listdir(inputs_dict['pose']) if fn.endswith('.npy')])\n"
    "        tgt_musk_path = os.path.join(inputs_dict['pose'], \"{}.npy\".format(index % pose_file_count))"
)
if old_pose_line not in content:
    raise RuntimeError(
        "patch_echomimic_pose_loop.py: не найдена ожидаемая строка 'tgt_musk_path = ...' — "
        "апстрим infer.py изменился, нужно сверить патч вручную"
    )
content = content.replace(old_pose_line, new_pose_block)

with open(INFER_PY_PATH, "w", encoding="utf-8") as f:
    f.write(content)

print("patch_echomimic_pose_loop.py: infer.py успешно пропатчен")
