import bentoml
import pandas as pd


class CirrhosisStatusModelRunner(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("cpu",)
    SUPPORTS_CPU_MULTI_THREADING = True

    def __init__(self, model: bentoml.Model) -> None:
        self.classifier = bentoml.sklearn.load_model(model)

    @bentoml.Runnable.method()
    def status_prediction(self, input_data: pd.DataFrame) -> pd.DataFrame:
        ids = input_data["id"]
        predict_probas = self.classifier.predict_proba(input_data)
        print(predict_probas)

        status_labels = ["Status_C", "Status_CL", "Status_D"]
        predictions = pd.DataFrame(
            {"id": ids, **dict(zip(status_labels, predict_probas.T))}
        )

        return predictions
