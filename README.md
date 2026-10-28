# 📰 Vietnamese Fake News Detection (CNN-SVM-Rule)

## 📌 Giới thiệu dự án

Dự án **Fake News Detection** nhằm xây dựng một hệ thống phát hiện tin giả tiếng Việt dựa trên mô hình lai CNN-SVM-Rule
Hệ thống bao gồm **toàn bộ pipeline AI hoàn chỉnh**, từ xử lý dữ liệu đến triển khai web:

* Phân tích dữ liệu (EDA)
* Tiền xử lý văn bản
* Huấn luyện mô hình 
* Suy luận & giải thích kết quả
* Triển khai Web App với FastAPI + Frontend

Ứng dụng cho phép người dùng **nhập nội dung tin tức** và nhận kết quả:

* Nhãn dự đoán (tin thật / tin giả )
* Độ tin cậy (confidence score)

---

## 🧠 Kiến trúc tổng thể

```
Dataset
   ↓
EDA (Exploratory Data Analysis)
   ↓
Preprocessing (Clean – Segment – Feature)
   ↓
Training 
   ↓
Inference + Explainability
   ↓
Web Application (FastAPI + UI)
```

---

## 📂 Cấu trúc thư mục chính

```
Fake_news_Classify/
├── checkpoints/          # Lưu trữ mô hình tối ưu (cnn_best.pt, svm_model.pkl)
├── dataset/              # Dữ liệu huấn luyện và kết quả EDA
├── json/                 # Cấu hình luật (rules_auto.json) 
├── preprocessing/        # Module tiền xử lý dữ liệu 
├── src/
│   ├── model/            # Kiến trúc mạng CNN (cnn_model.py)
│   ├── rules/            # Engine thực thi và tự động sinh luật 
│   ├── train/            # Script huấn luyện riêng biệt (train_cnn.py, train_svm.py)
│   ├── infer/            # Logic suy luận tích hợp (infer.py)
│   └── app/              # Cấu hình ứng dụng chính
├── web/                  # Backend và Logic điều hướng
├── templates/            # Giao diện người dùng 
├── result/               # Lưu lịch sử huấn luyện, đánh giá và feedback người dùng
├── main.py               # File thực thi khởi chạy toàn bộ hệ thống
└── requirements.txt      # Danh sách thư viện cần thiết
```

---

## 🌐 Chạy Web Application

Khởi động hệ thống web (Backend + Frontend):

```bash
python main/run_web.py
```

Sau đó mở trình duyệt và truy cập:

```
http://127.0.0.1:5500 (cho live server)
```


## 📈 Nơi lưu kết quả

| Thành phần          | Vị trí                        |
| ------------------- | ----------------------------- |
| Model đã huấn luyện | `checkpoints/`                |
| Dữ liệu đã xử lý    | `dataset/data_processed/`     |
| Lịch sử training    | `result/training_history.csv` |
| Feedback người dùng | `result/feedback/`            |

---

## 🚀 Công nghệ sử dụng

* CNN (Convolutional Neural Network)
* SVM (Support Vector Machine)
* Hệ thống luật (Rule-based system)
* HTML / CSS / JavaScript

---

## 📌 Hướng phát triển
* Tích hợp thêm nhiều mô hình phân loại
* Học tăng cường từ phản hồi người dùng
* Mở rộng phạm vi dữ liệu
* Triển khai Docker / Cloud

---