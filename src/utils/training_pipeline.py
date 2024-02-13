import mlflow
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder, FunctionTransformer, StandardScaler
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from xgboost import XGBClassifier

# sklearn output
from sklearn import set_config
set_config(transform_output="pandas")


def split_features(df, target):
    unique_values = df.nunique().drop(target)

    cat_features = unique_values[unique_values < 5].index.tolist()
    num_features = unique_values[unique_values >= 5].index.tolist()

    return unique_values, cat_features, num_features


def build_training_pipeline(df, target):
    # Custom Transformers
    # def diagnosis_date(X):
    #     X["Diagnosis_Date"] = X["Age"] - X["N_Days"]
    #     return X
    #
    # def age_years(X):
    #     X["Age_Years"] = round(X["Age"] / 365.25).astype("int16")
    #     return X

    # EXTRA_COLUMNS = ["Diagnosis_Date", "Age_Years"]
    _, CAT_FEATURES, NUM_FEATURES = split_features(df, target)

    # Encoders
    one_hot_encoder = ColumnTransformer([
        (
            "one_hot_encoder",
            OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
            ["Edema"]
        )
    ], remainder="drop", verbose_feature_names_out=False)

    ordinal_encoder = ColumnTransformer([
        (
            "ordinal_encoder",
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            [
                "Drug",
                "Sex",
                "Ascites",
                "Hepatomegaly",
                "Spiders",
                "Stage"
            ]
        )
    ], remainder="drop", verbose_feature_names_out=False)

    standard_scaler = ColumnTransformer([
        (
            "scaler",
            StandardScaler(),
            NUM_FEATURES
        )
    ], remainder="drop", verbose_feature_names_out=False)

    # Feature Enginering Pipeline
    # extra_features_pipeline = Pipeline([
    #     ("age_years", FunctionTransformer(diagnosis_date, validate=False)),
    #     ("diagnosis_date", FunctionTransformer(age_years, validate=False)),
    # ])

    feature_encoding_pipeline = Pipeline([
        (
            "features",
            FeatureUnion(
                [
                    ("scaler", standard_scaler),
                    ("ohe", one_hot_encoder),
                    ("ordinal", ordinal_encoder),
                ],
            ),
        )
    ])

    feature_engineering_pipeline = Pipeline([
        # ("extra_features", extra_features_pipeline),
        ("feature_encoding", feature_encoding_pipeline)
    ])

    # Machine learning model
    model = XGBClassifier(objective="multi_logloss")

    model_params = model.get_params()
    mlflow.log_params({f"model__{key}": value for key, value in model_params.items()})

    # Full pipeline
    model_pipeline = Pipeline([
        ("data-proccessing", clone(feature_engineering_pipeline)),
        ("model", model)
    ])

    return model_pipeline
