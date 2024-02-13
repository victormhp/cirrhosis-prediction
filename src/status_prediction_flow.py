from metaflow import FlowSpec, Parameter, current, step


class CirrhosisStatusFlow(FlowSpec):
    source_file = Parameter("source-file", help="Source CSV file")
    tracking_uri = Parameter("tracking-uri", help="MLflow tracking URI", default="http://127.0.0.1:5000")

    @step
    def start(self):
        import mlflow

        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment("/cirrhosis/StatusPrediction")
        run = mlflow.start_run()
        self.mlflow_run_id = run.info.run_id
        mlflow.set_tag("metaflow.runNumber", current.run_id)
        mlflow.set_tag("metaflow.flowName", current.flow_name)
        print(f"Primer paso – started run: {self.mlflow_run_id}")
        self.next(self.load_data)

    @step
    def load_data(self):
        import mlflow
        import pandas as pd
        from sklearn.preprocessing import LabelEncoder

        mlflow.set_tracking_uri(self.tracking_uri)
        with mlflow.start_run(run_id=self.mlflow_run_id):
            print("Loading Data")
            df_cirrhosis = pd.read_csv(self.source_file)

            # Remove id
            df_cirrhosis.drop(columns="id", inplace=True)

            # Encode labels
            label_encoder = LabelEncoder()
            labels = label_encoder.fit_transform(df_cirrhosis["Status"])
            df_cirrhosis["Status"] = labels

            self.df_cirrhosis = df_cirrhosis
            self.target = "Status"

        self.next(self.split_data)

    @step
    def split_data(self):
        import mlflow
        from sklearn.model_selection import train_test_split

        mlflow.set_tracking_uri(self.tracking_uri)
        with mlflow.start_run(run_id=self.mlflow_run_id):
            print("Splitting Data")
            original_count = len(self.df_cirrhosis)
            training_size = 0.60
            test_size = (1 - training_size) / 2

            training_count = int(original_count * training_size)
            test_count = int(original_count * test_size)
            # validation_count = original_count - training_count - test_count

            # Splitting data
            self.X = self.df_cirrhosis.drop(columns="Status")
            self.y = self.df_cirrhosis["Status"]

            X_train, X_rest, y_train, y_rest = train_test_split(self.X, self.y, train_size=training_count)
            X_test, X_valid, y_test, y_valid = train_test_split(X_rest, y_rest, train_size=test_count)

            mlflow.log_params(
                {
                    "dataset_size": original_count,
                    "training_set_size": len(X_train),
                    "validate_set_size": len(X_valid),
                    "test_set_size": len(X_test),
                }
            )

            self.X_train = X_train
            self.X_valid = X_valid
            self.X_test = X_test
            self.y_train = y_train
            self.y_valid = y_valid
            self.y_test = y_test

        self.next(self.model_training)

    @step
    def model_training(self):
        import mlflow
        from utils.training_pipeline import build_training_pipeline
        from utils.validate_model import validate_model

        mlflow.set_tracking_uri(self.tracking_uri)
        with mlflow.start_run(run_id=self.mlflow_run_id):
            print("Model Training")
            self.training_pipeline = build_training_pipeline(self.df_cirrhosis, self.target)
            self.training_pipeline.fit(self.X_train, self.y_train)
            train_score, valid_score = validate_model(self.training_pipeline, self.X_train, self.y_train)

            print("Train score - Mean Log Loss:", train_score)
            print("Validate score - Mean Log Loss:", valid_score)

            metrics_training = {
                "train_score": train_score,
                "validate_score": valid_score,
            }

            mlflow.log_metrics(metrics_training)

        self.next(self.model_validation)

    @step
    def model_validation(self):
        import mlflow
        from sklearn.metrics import log_loss

        mlflow.set_tracking_uri(self.tracking_uri)
        with mlflow.start_run(run_id=self.mlflow_run_id):
            print("Model Validation")
            y_pred = self.training_pipeline.predict_proba(self.X_valid)
            pred_score = log_loss(self.y_valid, y_pred)

            print("Train score - Mean Log Loss:", pred_score)
            metrics_validation = {
                "validation_score": pred_score,
            }
            mlflow.log_metrics(metrics_validation)

        self.next(self.model_registration)

    @step
    def model_registration(self):
        import mlflow
        import mlflow.sklearn

        mlflow.set_tracking_uri(self.tracking_uri)
        with mlflow.start_run(run_id=self.mlflow_run_id):
            print("Model Registration")
            mlflow.sklearn.log_model(self.training_pipeline, "cirrhosis-status-model")
            mlflow.register_model("runs:/{}/cirrhosis-status-model".format(self.mlflow_run_id), "cirrhosis-status-model")

        self.next(self.end)

    @step
    def end(self):
        print("<---- THE END ---->")
        pass


if __name__ == "__main__":
    CirrhosisStatusFlow()
