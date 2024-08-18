import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, ClientSettings
import numpy as np
import av

st.title("Keratitis Vision Simulator")

line_position = st.slider("Adjust Filter Position", min_value=0, max_value=100, value=50, step=1)

filter_options = ["Healthy Eye", "Early Stage", "Middle Stage", "Late Stage"]
filter = st.selectbox("Select Severity", filter_options, index=0)

st.logo("https://www.iapb.org/wp-content/uploads/2020/09/KeraLink-International.png", link="https://www.keralink.org/")

# Generate a fixed noise pattern that will be reused for every frame
def generate_fixed_noise_pattern(height, width):
    base_tint_color = 128
    noise_intensity = 10  # Adjust this to control how strong the variation is
    random_noise = np.random.randint(-noise_intensity, noise_intensity, (height, width, 3), dtype=np.int16)
    noise_pattern = np.clip(base_tint_color + random_noise, 0, 255).astype(np.uint8)
    return noise_pattern

def apply_filter_to_area(img, filter_type, noise_pattern):
    height, width, _ = img.shape

    # Define parameters for each filter type
    params = {
        "Early Stage": {"opacity": 0.25, "blur_radius": 31, "outer_blur_radius": 21},
        "Middle Stage": {"opacity": 0.2, "blur_radius": 71, "outer_blur_radius": 61},
        "Late Stage": {"opacity": 0.12, "blur_radius": 101, "outer_blur_radius": 91}
    }

    p = params.get(filter_type)

    # Apply Gaussian blur to the entire image
    blurred_img = cv2.GaussianBlur(img, (p["blur_radius"], p["blur_radius"]), 0)

    # Blend the blurred image with the fixed noise pattern using opacity
    tinted_blurred_img = cv2.addWeighted(blurred_img, p["opacity"], noise_pattern, 1 - p["opacity"], 0)

    return tinted_blurred_img

# Generate a fixed noise pattern once at the start (size will be set when the first frame is received)
noise_pattern = None

def transform(frame: av.VideoFrame):
    global noise_pattern

    img = frame.to_ndarray(format="bgr24")
    height, width, _ = img.shape

    # Generate the noise pattern if it hasn't been created yet
    if noise_pattern is None or noise_pattern.shape[:2] != (height, width):
        noise_pattern = generate_fixed_noise_pattern(height, width)

    split_point = int((line_position / 100) * width)
    left_half = img[:, :split_point]
    right_half = img[:, split_point:]

    if filter != "Healthy Eye":
        right_half = apply_filter_to_area(img, filter, noise_pattern)[:, split_point:]

    img = np.concatenate((left_half, right_half), axis=1)
    cv2.line(img, (split_point, 0), (split_point, height), (255, 255, 255), 3)

    return av.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_streamer(
    key="streamer",
    video_frame_callback=transform,
    sendback_audio=False,
    media_stream_constraints={
        "video": {"width": {"ideal": 320}, "height": {"ideal": 240}, "frameRate": {"ideal": 15}},
        "audio": False,
    },
    client_settings=ClientSettings(
        video_html_attrs={
            "playsinline": True,
            "controls": True,
            "muted": True,
        }
    )
)
