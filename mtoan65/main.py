import ffmpeg
import numpy as np
import tensorflow as tf
import os

import sys
# setting path
sys.path.append('../MyTransNetV2/')
from inference.transnetv2 import TransNetV2
from transnet_utils import draw_video_with_predictions, scenes_from_predictions


weights = "inference/transnetv2-weights"
vid_folder_path = "mtoan65/vids"
files = [os.path.join(vid_folder_path, f) for f in os.listdir(vid_folder_path) if os.path.isfile(os.path.join(vid_folder_path, f))]
files.remove(os.path.join(vid_folder_path, ".gitkeep"))
# w = [f for f in os.listdir(weights) if os.path.isfile(os.path.join(weights, f))]
visualize = True

print(files)
# print(w)

class TransNetParams:
    F = 16
    L = 3
    S = 2
    D = 256
    INPUT_WIDTH = 48
    INPUT_HEIGHT = 27
    CHECKPOINT_PATH = None

model = TransNetV2(weights)
for file in files:
    if os.path.exists(file + ".predictions.txt") or os.path.exists(
        file + ".scenes.txt"
    ):
        print(
            f"[TransNetV2] {file}.predictions.txt or {file}.scenes.txt already exists. "
            f"Skipping video {file}.",
            file=sys.stderr,
        )
        continue

    video_frames, single_frame_predictions, all_frame_predictions = model.predict_video(
        file
    )

    predictions = np.stack([single_frame_predictions, all_frame_predictions], 1)
    np.savetxt(file + ".predictions.txt", predictions, fmt="%.6f")

    scenes = model.predictions_to_scenes(single_frame_predictions)
    np.savetxt(file + ".scenes.txt", scenes, fmt="%d")

    # export video into numpy array using ffmpeg
    video_stream, err = (
        ffmpeg
        .input(file)
        .output('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(TransNetParams.INPUT_WIDTH, TransNetParams.INPUT_HEIGHT))
        .run(capture_stdout=True)
    )
    video = np.frombuffer(video_stream, np.uint8).reshape([-1, TransNetParams.INPUT_HEIGHT, TransNetParams.INPUT_WIDTH, 3])

    # For ilustration purposes, we show only 200 frames starting with frame number 8000.
    draw_video_with_predictions(video[8000:8200], predictions[8000:8200], threshold=0.1)

    if visualize:
        if os.path.exists(file + ".vis.png"):
            print(
                f"[TransNetV2] {file}.vis.png already exists. "
                f"Skipping visualization of video {file}.",
                file=sys.stderr,
            )
            continue

        pil_image = model.visualize_predictions(
            video_frames, predictions=(single_frame_predictions, all_frame_predictions)
        )
        pil_image.save(file + ".vis.png")
