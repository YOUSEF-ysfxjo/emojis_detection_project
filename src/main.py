import cv2
from deepface import DeepFace


EMOJI_MAP = {
    'happy' : ':)',
    'sad' : ':(',
    'angry' : '>:(',
    'surprise' : ':O',
    'fear' : 'D:',
    'disgust' : ':X',
    "neutral" : '._.'
}


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
            emoji = EMOJI_MAP.get(emotion, "??")
            emotion_scores = face_result['emotion']
            
            # Extract face region coordinates
            region = face_result['region']
            x, y, w, h = region['x'], region['y'], region['w'], region['h']

            if (h > 0) and (w > 0):
                     # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Put emotion text on frame
                text = f"{emotion} {emoji}"
                cv2.putText(
                    frame,
                    text,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2
                )
            
                emotions_data.append({
                    'dominant_emotion': emotion,
                    'emotion_scores': emotion_scores,
                    'face_region': {'x': x, 'y': y, 'w': w, 'h': h},
                    'emoji' : emoji
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
