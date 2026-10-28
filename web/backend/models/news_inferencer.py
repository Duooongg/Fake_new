# web/backend/models/news_inferencer.py

from src.infer.infer import NewsInferencer
import torch

class NewsInferencerWrapper:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = NewsInferencer()
        return cls._instance

    def infer(self, news_input: dict) -> dict:
        try:
            inferencer = self.get_instance()
            raw_result = inferencer.infer(news_input)

            pred = raw_result.get("prediction", {})

            def _to_float(x):
                if x is None: return 0.0
                if isinstance(x, torch.Tensor):
                    return float(x.item())
                return float(x)

            processed_output = {
                "prediction": {
                    "label": int(pred.get("label", 0)),
                    "confidence": _to_float(pred.get("confidence")),
                    "cnn_score": _to_float(pred.get("cnn_score")),
                    "svm_score": _to_float(pred.get("svm_score")),
                    "rule_score": _to_float(pred.get("rule_score")),
                },
                "explanation": raw_result.get("explanation", [])
            }
            
            print(" Data sent to Backend UI:", processed_output) # Log để kiểm tra
            return processed_output

        except Exception as e:
            print(f" ERROR in NewsInferencerWrapper: {str(e)}")
            raise