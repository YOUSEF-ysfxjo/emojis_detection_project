import cv2
import numpy as np
import requests
from deepface import DeepFace


EMOJI_MAP = {
    'happy': 'https://raw.githubusercontent.com/YOUSEF-ysfxjo/emojis_detection_project/main/images/happy.png',
    'sad': 'https://raw.githubusercontent.com/YOUSEF-ysfxjo/emojis_detection_project/main/images/sad.png',
    'angry': 'https://raw.githubusercontent.com/YOUSEF-ysfxjo/emojis_detection_project/main/images/angry.png',
    'surprise': 'https://raw.githubusercontent.com/YOUSEF-ysfxjo/emojis_detection_project/main/images/surprised.png',
    'fear': 'https://raw.githubusercontent.com/YOUSEF-ysfxjo/emojis_detection_project/main/images/afraid.png',
    'disgust': 'https://raw.githubusercontent.com/YOUSEF-ysfxjo/emojis_detection_project/main/images/disgusted.png',
    'neutral': 'https://raw.githubusercontent.com/YOUSEF-ysfxjo/emojis_detection_project/main/images/neutral.png'
}


def _load_emoji_images(map_dict):
    imgs = {}
    for k, path in map_dict.items():
        try:
            if isinstance(path, str) and path.startswith("http"):
                r = requests.get(path, timeout=5)
                arr = np.frombuffer(r.content, np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
            else:
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

            if img is None:
                imgs[k] = None
                continue

            # Normalize to BGRA
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
            elif img.shape[2] == 3:
                b, g, r = cv2.split(img)
                a = np.ones(b.shape, dtype=b.dtype) * 255
                img = cv2.merge((b, g, r, a))

            imgs[k] = img
        except Exception:
            imgs[k] = None
    return imgs


# load once
EMOJI_IMAGES = _load_emoji_images(EMOJI_MAP)


def _overlay_emoji(bg, fg, x, y, w, h):
    if fg is None:
        return bg

    try:
        # ensure positive integer size
        w = max(1, int(w))
        h = max(1, int(h))
        fg_resized = cv2.resize(fg, (w, h), interpolation=cv2.INTER_AREA)
    except Exception:
        return bg

    H, W = bg.shape[:2]
    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(W, x1 + w)
    y2 = min(H, y1 + h)

    if x1 >= x2 or y1 >= y2:
        return bg

    fx1 = x1 - int(x)
    fy1 = y1 - int(y)
    fx2 = fx1 + (x2 - x1)
    fy2 = fy1 + (y2 - y1)

    fg_region = fg_resized[fy1:fy2, fx1:fx2]
    if fg_region.size == 0:
        return bg

    bg_region = bg[y1:y2, x1:x2]

    if fg_region.shape[2] == 4:
        alpha = fg_region[:, :, 3] / 255.0
        for c in range(3):
            bg_region[:, :, c] = (alpha * fg_region[:, :, c] + (1 - alpha) * bg_region[:, :, c]).astype(bg_region.dtype)
    else:
        bg_region[:, :, :] = fg_region[:, :, :3]

    bg[y1:y2, x1:x2] = bg_region
    return bg


def process_frame(frame):

    #check if frame is empty
    if frame is None:
        metadata = {"error": "Empty frame"}
        return frame, metadata

    resolution = (640, 480)
    frame = cv2.resize(frame, resolution)
    #BGR to RGB

    metadata = {

    }

    return frame, metadata

def analysis(frame):
    """
    Analyze frame for emotions using DeepFace
    Returns frame with annotations and emotion data
    """
    
    if frame is None:
        return frame, {"error": "Empty frame"}

    rgb_frame = frame.copy()
    rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)
    
    try:
        # Analyze emotions in the frame
        result = DeepFace.analyze(
            img_path=rgb_frame,
            actions=['emotion'],
            enforce_detection=False
        )
        
        emotions_data = []
        
        # Process each detected face
        if isinstance(result, dict):
            result = [result]
        for face_result in result:
            emotion = face_result['dominant_emotion']
            emoji_url = EMOJI_MAP.get(emotion)
            emoji_img = EMOJI_IMAGES.get(emotion)
            emotion_scores = face_result['emotion']

            # Extract face region coordinates
            region = face_result.get('region', {})
            x, y, w, h = int(region.get('x', 0)), int(region.get('y', 0)), int(region.get('w', 0)), int(region.get('h', 0))

            if (h > 0) and (w > 0):
                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Overlay emoji image above or on the face
                # Place emoji so its bottom aligns with the top of the detected face
                emoji_w = w
                emoji_h = h
                emoji_x = x
                emoji_y = y - int(emoji_h * 0.6)
                if emoji_y < 0:
                    emoji_y = y

                frame = _overlay_emoji(frame, emoji_img, emoji_x, emoji_y, emoji_w, emoji_h)

                emotions_data.append({
                    'dominant_emotion': emotion,
                    'emotion_scores': emotion_scores,
                    'face_region': {'x': x, 'y': y, 'w': w, 'h': h},
                    'emoji': emoji_url
                })
            else:
                continue
        
        metadata = {
            'faces_detected': len(emotions_data),
            'emotions': emotions_data
        }
        
        return frame, metadata
        
    except Exception as e:
        return frame, {"error": str(e)}



if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        frame, meta1 = process_frame(frame)
        annotated_frame, meta2 = analysis(frame)

        
        cv2.imshow('Emotion Detection', annotated_frame)
        
        if 'faces_detected' in meta2:
            print(f"Faces detected: {meta2['faces_detected']}")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()

    cv2.destroyAllWindows()
