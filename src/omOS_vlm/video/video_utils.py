import cv2
from typing import List, Tuple

def enumerate_video_devices(index: int = 0) -> List[Tuple[int, str]]:
    devices = []
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            break
        devices.append((index, cap.getBackendName()))
        cap.release()
        index += 1
    return devices