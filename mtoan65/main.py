import ffmpeg
import numpy as np
import tensorflow as tf
import os

import sys
# setting path
sys.path.append('../MyTransNetV2/')
from inference.transnetv2 import TransNetV2


weights = "inference/transnetv2-weights"
vid_folder_path = "mtoan65/vids"
files = [os.path.join(vid_folder_path, f) for f in os.listdir(vid_folder_path) if os.path.isfile(os.path.join(vid_folder_path, f))]
files.remove(os.path.join(vid_folder_path, ".gitkeep"))
# w = [f for f in os.listdir(weights) if os.path.isfile(os.path.join(weights, f))]
visualize = True

print(files)
# print(w)

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
