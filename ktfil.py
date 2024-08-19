import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import numpy as np
import av
import psutil

st.title("Keratitis Vision Simulator")

line_position = st.slider("Adjust Filter Position", min_value=0, max_value=100, value=50, step=1)

filter_options = ["Healthy Eye", "Early Stage", "Middle Stage", "Late Stage"]
filter = st.selectbox("Select Severity", filter_options, index=0)

st.logo("https://www.iapb.org/wp-content/uploads/2020/09/KeraLink-International.png", link="https://www.keralink.org/")

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.noise_pattern = None

    def generate_fixed_noise_pattern(self, height, width):
        base_tint_color = 128
        noise_intensity = 5  # Adjust this to control how strong the variation is
        random_noise = np.random.randint(-noise_intensity, noise_intensity, (height, width, 3), dtype=np.int16)
        noise_pattern = np.clip(base_tint_color + random_noise, 0, 255).astype(np.uint8)
        return noise_pattern

    def apply_filter_to_area(self, img, filter_type):
        height, width, _ = img.shape

        # Generate the noise pattern at the correct size for the current frame
        noise_pattern = self.generate_fixed_noise_pattern(height, width)

        # Define parameters for each filter type
        params = {
            "Early Stage": {"opacity": 0.25, "blur_radius": 31, "outer_blur_radius": 21},
            "Middle Stage": {"opacity": 0.2, "blur_radius": 61, "outer_blur_radius": 51},
            "Late Stage": {"opacity": 0.15, "blur_radius": 91, "outer_blur_radius": 81}
        }

        p = params.get(filter_type)

        # Apply Gaussian blur to the image
        blurred_img = cv2.GaussianBlur(img, (p["blur_radius"], p["outer_blur_radius"]), 0)

        # Blend the blurred image with the generated noise pattern using opacity
        tinted_blurred_img = cv2.addWeighted(blurred_img, p["opacity"], noise_pattern, 1 - p["opacity"], 0)

        return tinted_blurred_img

    def transform(self, frame: av.VideoFrame):
        img = frame.to_ndarray(format="bgr24")
        height, width, _ = img.shape

        split_point = int((line_position / 100) * width)
        left_half = img[:, :split_point]
        right_half = img[:, split_point:]

        if filter != "Healthy Eye":
            # Apply filter to the right half of the image
            right_half = self.apply_filter_to_area(right_half, filter)

        img = np.concatenate((left_half, right_half), axis=1)
        cv2.line(img, (split_point, 0), (split_point, height), (255, 255, 255), 1)

        # Convert image back to VideoFrame
        video_frame = av.VideoFrame.from_ndarray(img, format="bgr24")
        
        # Clean up the image array to free memory
        del img

        return video_frame

def display_memory_usage():
    # Get the memory usage of the current process
    process = psutil.Process()
    mem_info = process.memory_info()
    # Convert memory usage from bytes to GB
    memory_usage_gb = mem_info.rss / (1024 ** 3)
    st.write(f"Memory usage: {memory_usage_gb:.2f} GB")

# Display memory usage on the Streamlit app
display_memory_usage()

# Use a simplified RTC configuration
rtc_config = RTCConfiguration({
    "iceServers": [{"urls": "stun:stun.l.google.com:19302"}]
})

webrtc_streamer(
    key="streamer",
    video_frame_callback=VideoProcessor().transform,
    rtc_configuration=rtc_config,
    media_stream_constraints={
        "video": {"width": {"ideal": 320}, "height": {"ideal": 240}, "frameRate": {"ideal": 30}},
        "audio": False,
    }
)
