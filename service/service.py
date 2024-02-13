import bentoml
import numpy as np
from bentoml.io import PandasDataFrame

from cirrhosis_status_runner import CirrhosisStatusModelRunner

MODEL_TAG = "cirrhosis-status-model"

cirrhosis_status_model = bentoml.sklearn.get(MODEL_TAG)
cirrhosis_status_model_runner = bentoml.Runner(
    CirrhosisStatusModelRunner,
    models=[cirrhosis_status_model],
    runnable_init_params={"model": cirrhosis_status_model},
)

cirrhosis_status_service = bentoml.Service("cirrhosis-status-service", runners=[cirrhosis_status_model_runner])


@cirrhosis_status_service.api(input=PandasDataFrame(), output=PandasDataFrame())
def predict(input_df):
    return cirrhosis_status_model_runner.status_prediction.run(input_df)
