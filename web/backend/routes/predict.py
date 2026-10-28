from fastapi import APIRouter, HTTPException
from web.backend.schemas import NewsInput
from web.backend.services.infer_service import InferService
from web.backend.services.result_mapper import map_result_to_ui
import traceback

router = APIRouter()

# =========================
# INIT SERVICE (SINGLETON)
# =========================
infer_service = InferService()

@router.post(
    "/predict",
    summary="Đánh giá tin tức",
    description="Nhận nội dung tin tức và trả về kết quả phân loại tin thật / tin giả"
)
def predict(news: NewsInput):
    """
    Controller layer:
    - Nhận request từ frontend (thông qua Pydantic NewsInput)
    - Gọi tầng service để chạy inference (CNN + SVM + Rules)
    - Mapping output sang định dạng UI mong muốn
    """

    try:
        #  Chuyển đổi Pydantic model sang dictionary
        input_data = news.model_dump()

        # Kiểm tra nội dung text đầu vào
        if "text" not in input_data or not input_data["text"].strip():
            raise ValueError("Trường 'text' là bắt buộc và không được để trống.")

        # 2️Gọi ML Core (Tầng Service)
        raw_model_output = infer_service.run_infer(input_data)

        # 3️⃣Map kết quả sang định dạng Frontend yêu cầu
        ui_response = map_result_to_ui(raw_model_output)

        # Log kiểm tra kết quả cuối cùng trước khi gửi đi
        print(" Final JSON sent to Frontend:", ui_response)

        return ui_response

    except ValueError as ve:
        # Lỗi dữ liệu đầu vào không hợp lệ (400)
        print(f" Validation Error: {str(ve)}")
        raise HTTPException(
            status_code=400,
            detail=str(ve)
        )

    except Exception as e:
        # Lỗi hệ thống hoặc lỗi trong quá trình inference (500)
        print(f" FATAL ERROR IN /predict: {str(e)}")
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=f"Internal error during prediction: {str(e)}"
        )