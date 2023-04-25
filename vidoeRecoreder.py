import cv2

# Set up video capture
cap = cv2.VideoCapture(0) # 0 is the default camera index
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = 30

# Set up video writer

fourcc = cv2.VideoWriter_fourcc(*'H264')
out = cv2.VideoWriter('output.mp4', fourcc, fps, (int(width), int(height)))

# Start recording
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # Do any image processing on the frame here
    out.write(frame)

    # Display the frame (optional)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
out.release()
cv2.destroyAllWindows()