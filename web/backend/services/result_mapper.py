from typing import Dict

LABEL_TEXT = {
    "fake_news": "Tin giả",
    "true_news": "Tin thật",
}

ID2LABEL = {
    0: "true_news",
    1: "fake_news"
}

# =========================
# UI HELPERS
# =========================

def _confidence_level(confidence: float) -> str:
    if confidence >= 0.75:
        return "high"
    if confidence >= 0.5:
        return "medium"
    return "low"

def _border_class(label: str) -> str:
    if label == "fake_news":
        return "CẢNH BÁO: Tin giả ⚠️"
    return "Tin đáng tin cậy ✅"

def _color_class(label: str) -> str:
    if label == "fake_news":
        return "text-red"
    return "text-green"

# =========================
# MAIN MAPPER
# =========================

def map_result_to_ui(model_output: Dict) -> Dict:
    """
    FINAL MAPPER (NO UNCERTAIN)
    - Tin giả hoặc Tin thật
    - UI KHÔNG tự quyết lại nhãn
    """

    pred = model_output.get("prediction", {})

    raw_label = int(pred.get("label", 0))
    confidence = float(pred.get("confidence", 0.0))

    cnn_score = float(pred.get("cnn_score", 0.0))
    svm_score = float(pred.get("svm_score", 0.0))
    rule_score = float(pred.get("rule_score", 0.0))
    decision_type = pred.get("decision_type", "unknown")

    final_label = ID2LABEL.get(raw_label, "true_news")

    summary_reasons = [
        f"• Điểm CNN (ngữ nghĩa & embedding): {cnn_score:.2f}",
        f"• Điểm SVM (đặc trưng CNN): {svm_score:.2f}",
        f"• Điểm Rule (luật ngôn ngữ): {rule_score:.2f}",
        f"• Cơ chế quyết định: {decision_type}"
    ]

    return {
        "status": "success",
        "result": {
            "final_label": final_label,
            "final_label_text": LABEL_TEXT[final_label],
            "confidence": round(confidence, 2),
            "confidence_level": _confidence_level(confidence),
            "cnn_score": round(cnn_score, 3),
            "svm_score": round(svm_score, 3),
            "rule_score": round(rule_score, 3),
            "decision_type": decision_type,
            "ui_state": {
                "border_class": _border_class(final_label),
                "color_class": _color_class(final_label)
            }
        },
        "top_predictions": [],
        "reasons": {
            "summary": summary_reasons
        },
        "explanation": model_output.get("explanation", {})
    }
