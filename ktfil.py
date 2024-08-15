import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer
import numpy as np
import av

st.title("Keratitis Vision Simulator")

slider_key = "split_position_slider"
line_position = st.slider("Adjust Filter Position", min_value=0, max_value=100, value=50, step=1, key=slider_key)

filter_options = ["Healthy Eye", "Early Stage", "Middle Stage", "Late Stage"]
filter = st.selectbox("Select Severity", filter_options, index=0)

video_placeholder = st.empty()

def apply_filter(img, filter_type):
    if filter_type in ["Early Stage", "Middle Stage", "Late Stage"]:
        img = circular_blur(img, filter_type)
    return img

def circular_blur(img, filter_type):
    height, width, _ = img.shape
    center = (width // 2, height // 2)

    if filter_type == "Early Stage":
        radius = min(width, height) // 5
        opacity = 0.2
        blur_radius = 31
        outer_blur_radius = 21
    elif filter_type == "Middle Stage":
        radius = min(width, height) // 3
        opacity = 0.1
        blur_radius = 71
        outer_blur_radius = 61
    elif filter_type == "Late Stage":
        radius = min(width, height) // 2
        opacity = 0.05
        blur_radius = 101
        outer_blur_radius = 91

    tint_color = (37, 47, 53)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)

    blurred_img = cv2.GaussianBlur(img, (blur_radius, blur_radius), 0)
    tinted_blurred_img = cv2.addWeighted(blurred_img, opacity, np.full_like(blurred_img, tint_color), 1 - opacity, 0)

    outer_blurred_img = cv2.GaussianBlur(img, (outer_blur_radius, outer_blur_radius), 0)

    img = np.where(mask[:, :, np.newaxis] == 255, tinted_blurred_img, outer_blurred_img)

    return img

def transform(frame: av.VideoFrame):
    img = frame.to_ndarray(format="bgr24")
    
    height, width, _ = img.shape
    slider_max = 100
    normalized_position = int((line_position / slider_max) * width)
    
    split_point = normalized_position

    left_half = img[:, :split_point]
    right_half = img[:, split_point:]

    if filter != "Healthy Eye":
        if filter in ["Early Stage", "Middle Stage", "Late Stage"]:
            blurred_img = circular_blur(img.copy(), filter)
            right_half = blurred_img[:, split_point:]
    
    img = np.concatenate((left_half, right_half), axis=1)

    line_thickness = 3
    cv2.line(img, (split_point, 0), (split_point, height), (255, 255, 255), line_thickness)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

with video_placeholder.container():
    webrtc_streamer(
        key="streamer",
        video_frame_callback=transform,
        sendback_audio=False,
        video_html_attrs={"playsinline": True, "controls": True}
    )
