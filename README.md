# GRAD_CAM-on-medical-image-of-bones
Sử dụng công cụ GRAD-CAM trên nhận dạng ảnh xương y khoa

# Ứng Dụng Grad-CAM Trong Nhận Dạng Ảnh Xương Gãy

## 1. Giới thiệu chung
Trong y tế và chẩn đoán hình ảnh, việc ứng dụng học sâu (Deep Learning) để tự động phát hiện các vùng xương gãy trên ảnh X-quang đang ngày càng phổ biến. Tuy nhiên, các mô hình mạng nơ-ron tích chập (CNN) thường được xem là các "hộp đen" (black-box), gây khó khăn trong việc giải thích lý do tại sao mô hình lại đưa ra một dự đoán cụ thể. 

Để giải quyết vấn đề này, kỹ thuật **Grad-CAM** (Gradient-weighted Class Activation Mapping) được sử dụng nhằm tạo ra các bản đồ kích hoạt lớp. Kỹ thuật này giúp trực quan hóa các vùng không gian trên ảnh đầu vào có ảnh hưởng lớn nhất đến quyết định phân loại của mô hình. Việc kết hợp Grad-CAM giúp tăng cường tính minh bạch và độ tin cậy của hệ thống chẩn đoán đối với các y bác sĩ.

## 2. Giới thiệu về công cụ Grad-CAM
Grad-CAM hoạt động bằng cách sử dụng thông tin đạo hàm (gradient) của một lớp mục tiêu truyền ngược vào tầng tích chập cuối cùng để tạo ra một bản đồ nhiệt (heatmap). Bản đồ nhiệt này làm nổi bật các khu vực trọng yếu. 

Hình ảnh dưới đây minh họa tổng quát về cách Grad-CAM và các biến thể của nó (như Grad-CAM++) tập trung vào các đặc trưng khác nhau của một hình ảnh để đưa ra dự đoán:

![Minh họa Grad-CAM]()
<img width="1438" height="564" alt="abcd" src="https://github.com/user-attachments/assets/9600e94f-ca6b-4f4e-ac1b-0ae57ffcba68" />

*Hình 1: Minh họa cách công cụ Grad-CAM và Grad-CAM++ làm nổi bật các vùng đặc trưng quan trọng trên một hình ảnh tổng quát.*

## 3. Kết quả Grad-CAM trên mô hình ResNet50 Fine-tune
Trong dự án này, chúng tôi tiến hành tinh chỉnh (fine-tune) kiến trúc mạng **ResNet50** để nhận dạng và khoanh vùng các vị trí xương gãy. Nhằm đánh giá khả năng "nhìn" và định vị của mô hình, kỹ thuật Grad-CAM và Grad-CAM++ được áp dụng trực tiếp lên các ảnh X-quang đầu vào.

Các kết quả dưới đây cho thấy mô hình ResNet50 đã học được các đặc trưng chính xác. Vùng có cường độ màu cao nhất (đỏ/vàng) trên bản đồ nhiệt trùng khớp với vị trí xương gãy thực tế (được đóng khung xanh lá - Ground Truth Bounding Box).

### Kết quả minh họa:

**Mẫu số 1:**
![Kết quả Grad-CAM mẫu 1](IMG0003703_gradcam.jpg)
<img width="1536" height="548" alt="IMG0003703_gradcam" src="https://github.com/user-attachments/assets/1c363894-944e-423f-9b2b-96f6e90560a7" />
*Hình 2: Trực quan hóa dự đoán trên mẫu IMG0003703. Bản đồ nhiệt bao phủ chính xác vị trí gãy nứt trên thân xương.*

**Mẫu số 2:**
![Kết quả Grad-CAM mẫu 2](IMG0003523_gradcam.jpg)
<img width="1536" height="548" alt="IMG0003523_gradcam" src="https://github.com/user-attachments/assets/217fd3ca-093e-47db-96f3-23b6901419b0" />
*Hình 3: Trực quan hóa dự đoán trên mẫu IMG0003523. Mô hình tập trung chính xác vào vùng tổn thương với xác suất dự đoán (Prob) lên tới 0.997.*

**Mẫu số 3:**
![Kết quả Grad-CAM mẫu 3](IMG0003686_gradcam.jpg)
<img width="1536" height="548" alt="IMG0003686_gradcam" src="https://github.com/user-attachments/assets/f8d53670-be12-491a-8f55-c2d749debdf7" />

*Hình 4: Trực quan hóa dự đoán trên mẫu IMG0003686. Cả Grad-CAM và Grad-CAM++ đều định vị thành công vùng xương gãy, hỗ trợ đắc lực cho việc chẩn đoán hình ảnh.*

## 4. Kết luận
Việc tích hợp Grad-CAM vào các hệ thống chẩn đoán xương gãy dựa trên ResNet50 không chỉ giúp xác minh tính chính xác của mô hình học sâu mà còn cung cấp bằng chứng trực quan, hỗ trợ các chuyên gia y tế đưa ra quyết định lâm sàng nhanh chóng và chính xác hơn.
