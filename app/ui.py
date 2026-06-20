import cv2
import gradio as gr
import yaml
from pathlib import Path
from src.core.model import ModelHandler
from src.core.processing import process_frame

# Load Config
config_path = Path("config/config.yaml")
if config_path.exists():
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
else:
    config = {
        "model": {"default_weights": "derived/yolov8n.pt"},
        "inference": {"conf_threshold": 0.3, "blur_intensity": 51},
    }

# Initialize Model Handler
model_handler = ModelHandler(trained_path=config["model"].get("trained_weights"))
# Pre-load model
model_handler.load_model()


def inference_image(image, conf_threshold, blur_intensity):
    if image is None:
        return None, None, "No image uploaded."

    # Process
    box_img, blur_img, inf_time = process_frame(
        model_handler, image, conf_threshold=conf_threshold, blur_intensity=int(blur_intensity)
    )

    # Convert BGR to RGB for Gradio (since OpenCV uses BGR internally, but gradio expects RGB/numpy arrays are passed as RGB)
    # Actually, Gradio passes images as RGB numpy arrays, so we don't need to convert input.
    # We output RGB arrays. OpenCV operations on numpy arrays don't care about channel order except for drawing functions.
    # cv2.rectangle and cv2.putText use BGR color tuples (255, 0, 0) -> Blue.
    # If image is RGB, (255,0,0) will be Red. This is fine, we want Red boxes anyway!

    time_str = f"Inference Time: {inf_time:.4f} seconds"
    return box_img, blur_img, time_str


def inference_video(video_path, conf_threshold, blur_intensity):
    if not video_path:
        return None

    output_path = "blurred_output.mp4"

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Frame is BGR from cv2
        _, blur_frame, _ = process_frame(
            model_handler, frame, conf_threshold=conf_threshold, blur_intensity=int(blur_intensity)
        )
        out.write(blur_frame)

    cap.release()
    out.release()
    return output_path


# Gradio Interface
with gr.Blocks(title="License Plate Blurring System") as demo:
    gr.Markdown("# 🚗 License Plate Detection & Blurring System")
    gr.Markdown(
        "Upload an image, video, or use your webcam to automatically detect and blur license plates using YOLOv8."
    )

    with gr.Row():
        conf_slider = gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=config["inference"]["conf_threshold"],
            step=0.05,
            label="Confidence Threshold",
        )
        blur_slider = gr.Slider(
            minimum=1,
            maximum=101,
            value=config["inference"]["blur_intensity"],
            step=2,
            label="Blur Intensity (Odd Number)",
        )

    with gr.Tabs():
        # IMAGE TAB
        with gr.Tab("Image / Webcam"):
            with gr.Row():
                with gr.Column():
                    img_input = gr.Image(
                        sources=["upload", "webcam"], type="numpy", label="Input Image"
                    )
                    img_btn = gr.Button("Process Image", variant="primary")
                with gr.Column():
                    inf_time_text = gr.Textbox(label="Performance", interactive=False)

            with gr.Row():
                img_box_output = gr.Image(label="Detections (Bounding Boxes)")
                img_blur_output = gr.Image(label="Blurred Result")

            img_btn.click(
                fn=inference_image,
                inputs=[img_input, conf_slider, blur_slider],
                outputs=[img_box_output, img_blur_output, inf_time_text],
            )

        # VIDEO TAB
        with gr.Tab("Video"):
            with gr.Row():
                vid_input = gr.Video(label="Input Video")
                vid_btn = gr.Button("Process Video", variant="primary")
            with gr.Row():
                vid_output = gr.Video(label="Blurred Video Output")

            vid_btn.click(
                fn=inference_video,
                inputs=[vid_input, conf_slider, blur_slider],
                outputs=[vid_output],
            )
