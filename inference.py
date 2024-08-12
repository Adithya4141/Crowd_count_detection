from ultralytics import YOLO
import cv2
import datetime
import sqlite3

conn = sqlite3.connect('people_data.db')
cur = conn.cursor()

# Load the YOLO model
model = YOLO("crowd_1.pt")

cur.execute('''CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    class TEXT NOT NULL,
    no INTEGER,
    Timestamp TEXT UNIQUE
)''')

# Initialize the video capture (0 for default webcam)
video_capture = cv2.VideoCapture(r"C:\Users\G.CHANDU\Downloads\airport_real time video.mp4")
frame_width = int(video_capture.get(3)) 
frame_height = int(video_capture.get(4)) 
   
size = (frame_width, frame_height) 

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 10.0,size)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    if not ret:
        break

    try:
        # Resize frame to the input size required by the model
        # image = cv2.resize(frame, (640, 640))

        # Perform prediction
        pred = model.predict(frame)

        # Extract detections
        detections = pred[0].boxes

        # Define the class you want to count (assuming class 'person' has index 0 in your model)
        target_class = 0  # Change this according to your model's class index for 'person'

        # Filter detections by the target class
        filtered_detections = [det for det in detections if int(det.cls) == target_class]

        # Count the number of objects
        object_count = len(filtered_detections)

        # Annotate each detected object with a label and number
        for i, det in enumerate(filtered_detections):
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            label = f"Person {i + 1}"
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        now = datetime.datetime.now()
        cur.execute('''INSERT INTO users (class, no, timestamp) VALUES ('humans', ?, ?)''', (object_count, now))
        conn.commit()

        # Write the frame to the video file
        out.write(frame)

        # Display the resulting frame with annotations
        cv2.imshow("Output", frame)

        # Wait for a key press (27 is the ESC key)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except cv2.error:
        pass

# Release the video capture and video write objects, and close windows
video_capture.release()
out.release()
cv2.destroyAllWindows()
conn.close()
