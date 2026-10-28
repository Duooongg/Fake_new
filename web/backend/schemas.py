from pydantic import BaseModel, Field
from typing import Optional, List, Any
from datetime import date

# ======================================================
# 1. INPUT SCHEMA (Frontend -> Backend)
# ======================================================

class NewsInput(BaseModel):
    title: Optional[str] = Field(None, description="Tiêu đề bài viết")
    author: Optional[str] = Field(None, description="Tác giả")
    text: str = Field(..., description="Nội dung chính")
    source: Optional[str] = Field(None, description="Nguồn tin")
    date_published: Optional[date] = Field(None, description="Ngày xuất bản")
    url: Optional[str] = Field(None, description="Đường dẫn bài viết")
    has_images: bool = Field(False)
    has_videos: bool = Field(False)
    num_shares: int = Field(0, ge=-1)
    num_comments: int = Field(0, ge=-1)

# ======================================================
# 2. UI RESPONSE COMPONENTS
# ======================================================

class UIState(BaseModel):
    border_class: str
    color_class: str

class FinalResult(BaseModel):
    final_label: str
    final_label_text: str
    confidence: float
    confidence_level: str
    cnn_score: Optional[float] = 0.0
    svm_score: Optional[float] = 0.0
    rule_score: Optional[float] = 0.0
    ui_state: UIState

class Reasons(BaseModel):
    summary: List[str]

# ======================================================
# 3. UI RESPONSE (Backend -> Frontend)
# ======================================================

class UIResponse(BaseModel):
    status: str
    result: FinalResult
    explanation: Optional[List[Any]] = [] 
    top_predictions: List[Any] = []
    reasons: Reasons

# ======================================================
# 4. FEEDBACK SCHEMA
# ======================================================

class UserFeedback(BaseModel):
    score: int = Field(..., ge=1, le=5)
    comment: Optional[str] = None

class FeedbackRequest(BaseModel):
    news_input: NewsInput
    model_output: UIResponse
    user_feedback: UserFeedback