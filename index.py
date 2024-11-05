import cv2
import tensorflow as tf
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import numpy as np
import base64  # Add this lin
import cv2
app = Flask(__name__)
socketio = SocketIO(app)

# Load the TensorFlow model
model = tf.keras.models.load_model('survellance_model.keras')


import cv2
import numpy as np

def preprocess(frame):
    # Resize the frame to the expected size of the model (128, 128, 3)
    frame_resized = cv2.resize(frame, (128, 128))
    # Normalize if necessary (e.g., dividing by 255 to scale pixel values to [0, 1])
    frame_normalized = frame_resized / 255.0
    # Expand dimensions to match the expected input shape for prediction (1, 128, 128, 3)
    return np.expand_dims(frame_normalized, axis=0)

sz = 128  # Resize dimensions to match model's input shape

def preprocess(frame):
    frame_resized = cv2.resize(frame, (sz, sz))  # Resize to (128, 128)
    frame_normalized = frame_resized / 255.0
    return np.expand_dims(frame_normalized, axis=0)


import logging
logging.basicConfig(level=logging.DEBUG)

def preprocess(frame):
    frame_resized = cv2.resize(frame, (sz, sz))  # Resize to (128, 128)
    frame_normalized = frame_resized / 255.0
    return np.expand_dims(frame_normalized, axis=0)

@socketio.on('start_stream')
def start_stream():
    video_capture = cv2.VideoCapture(0)  # Replace 0 with video path if using a file

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break
        labels=['Not Stealing', 'Stealing']
        # Process and classify the frame
        preprocessed_frame = preprocess(frame)
        prediction = model.predict(preprocessed_frame)

        # Get the class with the highest probability
        predicted_class = np.argmax(prediction, axis=1)[0]
        label=labels[predicted_class]
        # Overlay prediction text on the frame
        text = f"Prediction: Results {label}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_encoded = base64.b64encode(buffer).decode('utf-8')

        # Emit both frame and prediction
        socketio.emit('classification_result', {
            'image': frame_encoded,
            'prediction': str(label)
        })

        socketio.sleep(0.1)  # Adjust the sleep interval to control frame rate

    video_capture.release()




def video_stream():
    # Capture video from a webcam or video file
    cap = cv2.VideoCapture(0)  # 0 for webcam; or provide a video path

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Resize and preprocess for your model
        prediction = process_frame(frame)
        # Send result to frontend
        socketio.emit('classification_result', prediction.tolist())

    cap.release()




# @socketio.on('start_stream')
# def start_stream():
#     video_capture = cv2.VideoCapture(0)  # Use 0 for default webcam

#     while video_capture.isOpened():
#         ret, frame = video_capture.read()
#         if not ret:
#             break

#         # Process and classify the frame
#         prediction = process_frame(frame)

#         # Send prediction result to the frontend
#         socketio.emit('classification_result', {'prediction': prediction.tolist()})

#         # Optional: Add a delay for lower CPU usage
#         socketio.sleep(0.1)  # Adjust as needed for performance
    
#     video_capture.release()


@socketio.on('connect')
def handle_connect():
    print("Client connected")

@socketio.on('disconnect')
def handle_disconnect():
    print("Client disconnected")

# @app.route('/')
# def index():
#     return render_template('index.html')


@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    socketio.start_background_task(target=video_stream)
    socketio.run(app, debug=True)
