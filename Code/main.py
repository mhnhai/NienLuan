import cv2
import numpy as np
from ultralytics import YOLO
import os

# Hàm chuyển đổi pixel sang mét
def pixel_to_meters(pixel_length, reference_length_pixels, reference_length_meters):
    try:
        return (pixel_length * reference_length_meters) / reference_length_pixels
    except ZeroDivisionError:
        print("Lỗi: reference_length_pixels không được bằng 0!")
        return 0

# Hàm xử lý hình ảnh và đo kích thước xe
def detect_and_measure_car(image_path, reference_length_pixels, reference_length_meters):
    # Kiểm tra file ảnh tồn tại
    if not os.path.exists(image_path):
        print(f"Lỗi: Không tìm thấy file ảnh tại {image_path}")
        return

    # Đọc hình ảnh
    img = cv2.imread(image_path)
    if img is None:
        print(f"Lỗi: Không thể đọc hình ảnh từ {image_path}. Vui lòng kiểm tra định dạng hoặc đường dẫn.")
        return

    # Tải mô hình YOLOv8
    try:
        model = YOLO('yolov8x.pt')  # Sử dụng yolov8s.pt (nhỏ, cân bằng tốc độ và độ chính xác)
    except Exception as e:
        print(f"Lỗi khi tải mô hình YOLOv8: {str(e)}")
        return

    # Nhận dạng xe bằng YOLOv8
    try:
        results = model(img, verbose=False)  # Dự đoán trên hình ảnh, tắt log chi tiết
    except Exception as e:
        print(f"Lỗi khi chạy YOLOv8: {str(e)}")
        return

    # Kiểm tra xem có phát hiện đối tượng nào không
    if not results or len(results) == 0:
        print("Không phát hiện được đối tượng nào trong hình ảnh!")
        return

    # Lấy kết quả từ YOLOv8
    car_detected = False
    for result in results:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            print("Không có hộp bao nào được phát hiện!")
            continue

        for box in boxes:
            if result.names[int(box.cls)] == 'car' and box.conf > 0.3:  # Lọc xe với độ tin cậy > 0.5
                car_detected = True
                # Lấy tọa độ hộp bao
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                # Tính chiều dài và chiều rộng (theo pixel)
                width_pixels = x2 - x1
                height_pixels = y2 - y1

                # Chuyển đổi sang mét
                width_meters = pixel_to_meters(width_pixels, reference_length_pixels, reference_length_meters)
                length_meters = pixel_to_meters(height_pixels, reference_length_pixels, reference_length_meters)

                # Vẽ hộp bao quanh xe
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                # Ghi kích thước lên ảnh
                label = f"L: {length_meters:.2f}m, W: {width_meters:.2f}m"
                cv2.putText(img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                print(f"Xe phát hiện: Chiều dài = {length_meters:.2f}m, Chiều rộng = {width_meters:.2f}m")

    if not car_detected:
        print("Không phát hiện được xe nào trong hình ảnh với độ tin cậy > 0.5!")

    # Hiển thị hình ảnh với kết quả
    cv2.imshow("Kết quả nhận dạng", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Lưu hình ảnh kết quả (tùy chọn)
    try:
        cv2.imwrite("result_image.jpg", img)
        print("Hình ảnh kết quả đã được lưu tại result_image.jpg")
    except Exception as e:
        print(f"Lỗi khi lưu hình ảnh: {str(e)}")

# Tham số đầu vào
image_path = "car_image.jpg"  # Đường dẫn đến hình ảnh top view của xe
reference_length_pixels = 100  # Độ dài tham chiếu (pixel) của một vật thể trong ảnh
reference_length_meters = 1.0  # Độ dài thực tế (mét) của vật thể tham chiếu

# Gọi hàm xử lý
detect_and_measure_car(image_path, reference_length_pixels, reference_length_meters)