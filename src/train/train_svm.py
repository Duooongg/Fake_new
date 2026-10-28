import os
import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

# ======================================================
# CONFIG
# ======================================================
FEATURE_DIR = "src/features/cnn_features"
MODEL_SAVE_PATH = "checkpoints/svm_model.pkl"

os.makedirs("checkpoints", exist_ok=True)

# ======================================================
# LOAD EXTRACTED FEATURES (.npy)
# ======================================================
def load_npy_features():
    print(f"📥 Đang tải đặc trưng từ {FEATURE_DIR}...")
    
    try:
        X_train = np.load(os.path.join(FEATURE_DIR, "X_train.npy"))
        y_train = np.load(os.path.join(FEATURE_DIR, "y_train.npy"))
        
        X_test = np.load(os.path.join(FEATURE_DIR, "X_test.npy"))
        y_test = np.load(os.path.join(FEATURE_DIR, "y_test.npy"))
        
        return X_train, y_train, X_test, y_test
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Không tìm thấy file: {e.filename}. Hãy chạy extract_cnn_features.py trước!")

# ======================================================
# TRAIN SVM PIPELINE
# ======================================================
def main():
    # 1. Tải dữ liệu
    try:
        X_train, y_train, X_test, y_test = load_npy_features()
        print(f" Đã tải! Train: {X_train.shape}, Test: {X_test.shape}")
    except Exception as e:
        print(str(e))
        return

    # 2. Khởi tạo Pipeline (Scaler + SVM)
    # StandardScaler: Cực kỳ quan trọng để SVM không bị lệch điểm (bias)
    # kernel='rbf': Giúp xử lý các đặc trưng phi tuyến tính phức tạp từ CNN
    print(" Đang huấn luyện SVM Pipeline (StandardScaler + RBF Kernel)...")
    
    svm_pipeline = Pipeline([
        ('scaler', StandardScaler()), 
        ('svm', SVC(
            kernel="rbf",           # Chuyển từ linear sang rbf để bắt đặc trưng tốt hơn
            C=1.0,                  # Tham số Regularization
            gamma='scale',          # Tự động tính toán hệ số gamma
            probability=True,       # Bắt buộc để dùng được predict_proba trong infer.py
            class_weight="balanced", # Xử lý nếu tập dữ liệu bị lệch (mất cân bằng nhãn)
            verbose=True
        ))
    ])
    
    # 3. Huấn luyện
    svm_pipeline.fit(X_train, y_train)

    # 4. Đánh giá mô hình
    print("\n ĐÁNH GIÁ MÔ HÌNH SVM:")
    y_pred = svm_pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"Tổng độ chính xác (Accuracy): {acc:.4f}")
    print("-" * 30)
    print(classification_report(y_test, y_pred, target_names=["Tin Thật (True)", "Tin Giả (Fake)"]))

    # 5. Lưu toàn bộ Pipeline
    # Khi lưu Pipeline, Scaler cũng được lưu kèm để dùng cho dữ liệu thực tế sau này
    joblib.dump(svm_pipeline, MODEL_SAVE_PATH)
    print(f"\n Đã lưu SVM Pipeline thành công tại → {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()