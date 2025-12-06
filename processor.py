import cv2
from deepface import DeepFace


def process_frame(frame):

    #check if frame is empty
    if frame is None:
        metadata = {"error": "Empty frame"}
        return frame, metadata

    resolution = (640, 480)
    frame = cv2.resize(frame, resolution)
    #BGR to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


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
    
    try:
        # Analyze emotions in the frame
        result = DeepFace.analyze(
            img_path=frame,
            actions=['emotion'],
            enforce_detection=False
        )
        
        emotions_data = []
        
        # Process each detected face
        for face_result in result:
            emotion = face_result['dominant_emotion']
            emotion_scores = face_result['emotion']
            
            # Extract face region coordinates
            region = face_result['region']
            x, y, w, h = region['x'], region['y'], region['w'], region['h']
            
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Put emotion text on frame
            text = f"{emotion}: {emotion_scores[emotion]:.2f}"
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
                'face_region': {'x': x, 'y': y, 'w': w, 'h': h}
            })
        
        metadata = {
            'faces_detected': len(result),
            'emotions': emotions_data
        }
        
        return frame, metadata
        
    except Exception as e:
        return frame, {"error": str(e)}

