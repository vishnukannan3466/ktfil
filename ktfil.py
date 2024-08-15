import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer
import numpy as np
import av

st.title("Keratitis Vision Simulator")

line_position = st.slider("Adjust Filter Position", min_value=0, max_value=100, value=50, step=1)

filter_options = ["Healthy Eye", "Early Stage", "Middle Stage", "Late Stage"]
filter = st.selectbox("Select Severity", filter_options, index=0)

def circular_blur(img, filter_type):
    height, width, _ = img.shape
    center = (width // 2, height // 2)

    params = {
        "Early Stage": {"radius": min(width, height) // 5, "opacity": 0.2, "blur_radius": 31, "outer_blur_radius": 21},
        "Middle Stage": {"radius": min(width, height) // 3, "opacity": 0.1, "blur_radius": 71, "outer_blur_radius": 61},
        "Late Stage": {"radius": min(width, height) // 2, "opacity": 0.05, "blur_radius": 101, "outer_blur_radius": 91}
    }

    p = params.get(filter_type)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, center, p["radius"], 255, -1)

    blurred_img = cv2.GaussianBlur(img, (p["blur_radius"], p["blur_radius"]), 0)
    tinted_blurred_img = cv2.addWeighted(blurred_img, p["opacity"], np.full_like(blurred_img, (37, 47, 53)), 1 - p["opacity"], 0)
    outer_blurred_img = cv2.GaussianBlur(img, (p["outer_blur_radius"], p["outer_blur_radius"]), 0)

    return np.where(mask[:, :, np.newaxis] == 255, tinted_blurred_img, outer_blurred_img)

def transform(frame: av.VideoFrame):
    img = frame.to_ndarray(format="bgr24")
    height, width, _ = img.shape

    split_point = int((line_position / 100) * width)
    left_half = img[:, :split_point]
    right_half = img[:, split_point:]

    if filter != "Healthy Eye":
        right_half = circular_blur(img, filter)[:, split_point:]

    img = np.concatenate((left_half, right_half), axis=1)
    cv2.line(img, (split_point, 0), (split_point, height), (255, 255, 255), 3)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key="streamer",
    video_frame_callback=transform,
    sendback_audio=False,
    video_html_attrs={"playsinline": True, "controls": True}
)
