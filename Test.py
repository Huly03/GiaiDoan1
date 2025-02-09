import cv2
import os
from ultralytics import YOLO
from tkinter import filedialog
from tkinter import Tk

# Đường dẫn ảnh đầu vào
image_path = 'D:\person.jpg'  # Thay bằng đường dẫn ảnh của bạn

# Load mô hình YOLOv8
model = YOLO("yolov8n.pt")  # YOLOv8 nano (nhẹ, nhanh)

# Dự đoán trên ảnh
results = model(image_path)

# Đọc ảnh vào
img = cv2.imread(image_path)

# Dự đoán bounding boxes
for box in results[0].boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Lấy tọa độ bounding box
    conf = box.conf[0]  # Độ chính xác
    cls = int(box.cls[0])  # Nhãn dự đoán
    label = model.names[cls]  # Lấy tên đối tượng

    # Chỉ hiển thị nếu là "person"
    if label == "person":
        # Vẽ bounding box và nhãn
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Vẽ khung xanh
        cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Hiển thị ảnh
cv2.imshow("Detected Person", img)

# Mở hộp thoại lưu ảnh nếu người dùng nhấn phím "s" để lưu ảnh
key = cv2.waitKey(0)

if key == ord('s'):  # Nếu nhấn "s" thì lưu ảnh
    # Mở hộp thoại chọn thư mục lưu ảnh
    root = Tk()
    root.withdraw()  # Ẩn cửa sổ chính của Tkinter
    file_path = filedialog.asksaveasfilename(defaultextension=".jpg",
                                             filetypes=[("JPEG files", "*.jpg"), ("All files", "*.*")],
                                             title="Save Image As")

    if file_path:  # Nếu người dùng chọn thư mục và tên tệp
        cv2.imwrite(file_path, img)

cv2.destroyAllWindows()
