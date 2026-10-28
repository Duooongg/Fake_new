import json
import os
from datetime import datetime
from uuid import uuid4


class FeedbackService:
    def __init__(self, base_path="result/feedback"):
        self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)

    def save_feedback(self, full_data: dict):
        """
        Lưu feedback gộp theo ngày:
        result/feedback/YYYY_MM_DD.json
        """

        # ===== DATE INFO
        now = datetime.now()
        date_str = now.strftime("%Y_%m_%d")
        time_str = now.strftime("%H:%M:%S")

        file_path = os.path.join(
            self.base_path,
            f"{date_str}.json"
        )

        # ===== LOAD EXISTING DATA
        data_list = []
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data_list = json.load(f)
                    if not isinstance(data_list, list):
                        data_list = []
            except Exception:
                data_list = []

        # ===== BUILD FEEDBACK RECORD (SAFE COPY)
        feedback_record = {
            "feedback_id": str(uuid4()),
            "date": date_str,
            "time": time_str,
            "data": full_data
        }

        data_list.append(feedback_record)

        # ===== SAVE
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(
                    data_list,
                    f,
                    ensure_ascii=False,
                    indent=2
                )

            return {
                "status": "success",
                "file": f"{date_str}.json",
                "message": "Feedback saved successfully"
            }

        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
