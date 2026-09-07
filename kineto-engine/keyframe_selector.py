#!/usr/bin/env python3
"""
Kineto Keyframe Selector - Gradio UI
拖动进度条浏览视频，手动捕获 4 个康复关键帧，导出 keyframes_4.json
"""

import argparse
import json
from pathlib import Path

import cv2
import gradio as gr
import numpy as np


def load_pose_data(pose_data_path):
    with open(pose_data_path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_ui(video_path, pose_data_path, output_dir="output"):
    pose_data = load_pose_data(pose_data_path)
    keyframes_data = pose_data["keyframes"]
    total_frames = pose_data["metadata"]["total_frames"]
    fps = pose_data["metadata"]["video_fps"]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    captured = [None, None, None, None]

    def get_frame_at(frame_idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ret, frame = cap.read()
        if not ret:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def get_frame_info(frame_idx):
        idx = int(frame_idx)
        if 0 <= idx < len(keyframes_data):
            kf = keyframes_data[idx]
            return {
                "frame_index": kf["frame_index"],
                "timestamp_ms": kf["timestamp_ms"],
                "joints_3d_count": len(kf["joints_3d"]),
                "smpl_thetas_count": len(kf["smpl_thetas"]),
                "confidence": kf.get("confidence_score", 0),
                "cam_t": kf.get("cam_t", [0, 0, 0]),
            }
        return {"error": "frame out of range"}

    def capture_keyframe(slot, frame_idx):
        slot = int(slot)
        if slot < 1 or slot > 4:
            return "槽位必须是 1-4"
        idx = int(frame_idx)
        captured[slot - 1] = idx
        status = " | ".join(
            [f"帧{captured[i]}" if captured[i] is not None else "空" for i in range(4)]
        )
        return f"关键帧 {slot} = 帧 {idx}  |  [{status}]"

    def export_keyframes():
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        result = []
        for i, frame_idx in enumerate(captured):
            if frame_idx is None:
                continue
            if 0 <= frame_idx < len(keyframes_data):
                kf = keyframes_data[frame_idx]
                result.append({
                    "slot": i + 1,
                    "frame_index": kf["frame_index"],
                    "timestamp_ms": kf["timestamp_ms"],
                    "joints_3d": kf["joints_3d"],
                    "smpl_thetas": kf["smpl_thetas"],
                    "cam_t": kf.get("cam_t", [0, 0, 0]),
                    "confidence_score": kf.get("confidence_score", 0),
                })

        json_path = out_path / "keyframes_4.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"keyframes": result}, f, indent=2, ensure_ascii=False)

        return f"已导出 {len(result)} 个关键帧 → {json_path}"

    def update_display(frame_idx):
        img = get_frame_at(frame_idx)
        if img is None:
            return None, {"error": "cannot read frame"}
        info = get_frame_info(frame_idx)
        return img, info

    with gr.Blocks(title="Kineto 关键帧选择器") as demo:
        gr.Markdown("# Kineto 康复关键帧选择器")
        gr.Markdown("拖动进度条浏览视频，点击按钮捕获 4 个关键帧，最后导出 keyframes_4.json")

        with gr.Row():
            with gr.Column(scale=2):
                frame_display = gr.Image(label="视频帧", type="numpy")
                frame_slider = gr.Slider(
                    minimum=0, maximum=max(total_frames - 1, 1),
                    step=1, value=0, label="帧位置")
                frame_info = gr.JSON(label="当前帧参数")

            with gr.Column(scale=1):
                gr.Markdown("### 捕获关键帧")
                status_text = gr.Textbox(label="捕获状态", interactive=False)
                with gr.Row():
                    for i in range(1, 5):
                        gr.Button(f"捕获 #{i}").click(
                            fn=lambda f, s=i: capture_keyframe(s, f),
                            inputs=frame_slider, outputs=status_text)
                export_btn = gr.Button("导出 keyframes_4.json", variant="primary")
                export_text = gr.Textbox(label="导出结果", interactive=False)

        frame_slider.change(
            fn=update_display,
            inputs=frame_slider,
            outputs=[frame_display, frame_info])

        export_btn.click(fn=export_keyframes, inputs=None, outputs=export_text)

    cap.release()
    return demo


def main():
    parser = argparse.ArgumentParser(description="Kineto 关键帧选择器")
    parser.add_argument("--video", "-v", default="input_video.mp4", help="输入视频路径")
    parser.add_argument("--pose-data", "-p", default="output/pose_data.json", help="pose_data.json 路径")
    parser.add_argument("--output", "-o", default="output", help="输出目录")
    parser.add_argument("--port", type=int, default=7860, help="服务端口")
    args = parser.parse_args()

    demo = create_ui(args.video, args.pose_data, args.output)
    demo.launch(server_port=args.port)


if __name__ == "__main__":
    main()
