# Real-Time ASL Alphabet Interpreter
A Python application that translates American Sign Language (ASL) alphabet gestures into text in real-time using a webcam.

## 🛠️ Tech Stack
* **Python**
* **OpenCV**
* **MediaPipe**

## 🚀 How It Works
1.  The application uses **OpenCV** to capture video from the user's webcam.
2.  Each frame is processed by **MediaPipe's** `solutions.hands` model, which detects and returns the (x, y, z) coordinates of 21 key landmarks on the hand.
3.  The coordinates of these landmarks are collected and (Mô tả logic của bạn ở đây: ví dụ: "fed into a trained Machine Learning model (SVM/KNN)" hoặc "analyzed using geometric calculations") to classify the gesture.
4.  The predicted letter is then displayed on the GUI, providing instant feedback.
