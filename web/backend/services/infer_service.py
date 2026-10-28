# web/backend/services/infer_service.py

from typing import Dict
from web.backend.models.news_inferencer import NewsInferencerWrapper

class InferService:
    """
    Service layer:
    - Nhận input từ Controller (predict.py)
    - Gọi NewsInferencerWrapper để chạy AI Core (CNN + SVM + Rules)
    - Trả về dữ liệu gốc (raw output) để Controller tự Mapping
    """

    def __init__(self):
        # Sử dụng get_instance() để đảm bảo model chỉ được load 1 lần duy nhất (Singleton)
        self.inferencer = NewsInferencerWrapper.get_instance()

    def run_infer(self, news_input: Dict) -> Dict:
        """
        Thực hiện dự đoán tin tức từ AI Core
        """
        # Kiểm tra đầu vào
        if not isinstance(news_input, dict):
            raise ValueError("Dữ liệu đầu vào phải là một dictionary (JSON object)")

        # Log đầu vào để debug (Kiểm tra xem text đã được truyền tới đây chưa)
        print(f" [InferService] Đang xử lý tin tức...")

        # ======================================================
        # 1️ CHẠY MÔ HÌNH AI (CNN -> SVM -> RULE ENGINE)
        # Trả về: {'prediction': {...}, 'explanation': [...]}
        # ======================================================
        try:
            raw_output = self.inferencer.infer(news_input)
            
            # Kiểm tra xem AI Core có trả về đúng format không
            if not isinstance(raw_output, dict) or "prediction" not in raw_output:
                print(" [InferService] Cảnh báo: AI Core trả về dữ liệu thiếu trường 'prediction'")
            
            # Log kết quả thô để kiểm tra trong Terminal
            print(" [InferService] AI Core đã chạy xong.")
            
            return raw_output

        except Exception as e:
            print(f" [InferService] Lỗi khi chạy AI Core: {str(e)}")
            raise e

