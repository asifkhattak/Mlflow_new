
import warnings
import sys
import logging

import pandas as pd
import numpy as np

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

import dagshub

dagshub.init(
   dagshub.init(repo_owner='asifkhattak333', 
                repo_name='Mlflow_new',
                  mlflow=True)

)




# ============================================================
# Logging Configuration
# ============================================================

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


# ============================================================
# Evaluation Function
# ============================================================

def eval_metrics(actual, pred):
    """
    Calculate regression evaluation metrics.
    """

    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)

    return rmse, mae, r2


# ============================================================
# Main Program
# ============================================================

if __name__ == "__main__":

    warnings.filterwarnings("ignore")

    # Reproducibility
    np.random.seed(40)


    # ========================================================
    # 1. Load Dataset
    # ========================================================

    csv_url = (
        "https://dagshub.com/asifkhattak333/Mlflow_new.mlflow"
        "master/tests/datasets/winequality-red.csv"
    )

    try:

        data = pd.read_csv(
            csv_url,
            sep=";"
        )

        print("Dataset loaded successfully.")
        print(f"Dataset shape: {data.shape}")

    except Exception as e:

        logger.exception(
            "Unable to download training/test CSV. "
            "Check your internet connection. Error: %s",
            e
        )

        sys.exit(1)


    # ========================================================
    # 2. Train/Test Split
    # ========================================================

    train, test = train_test_split(
        data,
        test_size=0.25,
        random_state=42
    )


    # ========================================================
    # 3. Separate Features and Target
    # ========================================================

    train_x = train.drop(
        ["quality"],
        axis=1
    )

    test_x = test.drop(
        ["quality"],
        axis=1
    )

    train_y = train[["quality"]]

    test_y = test[["quality"]]


    # ========================================================
    # 4. Read Hyperparameters
    # ========================================================

    alpha = (
        float(sys.argv[1])
        if len(sys.argv) > 1
        else 0.5
    )

    l1_ratio = (
        float(sys.argv[2])
        if len(sys.argv) > 2
        else 0.5
    )


    # ========================================================
    # 5. Start MLflow Run
    # ========================================================

    with mlflow.start_run():

        print("\nMLflow run started.")

        # ----------------------------------------------------
        # Create Model
        # ----------------------------------------------------

        lr = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            random_state=42
        )


        # ----------------------------------------------------
        # Train Model
        # ----------------------------------------------------

        lr.fit(
            train_x,
            train_y
        )


        # ----------------------------------------------------
        # Make Predictions
        # ----------------------------------------------------

        predicted_qualities = lr.predict(
            test_x
        )


        # ----------------------------------------------------
        # Calculate Metrics
        # ----------------------------------------------------

        rmse, mae, r2 = eval_metrics(
            test_y,
            predicted_qualities
        )


        # ====================================================
        # 6. Display Results
        # ====================================================

        print(
            "\nElasticNet model "
            "(alpha={:f}, l1_ratio={:f}):"
            .format(alpha, l1_ratio)
        )

        print("  RMSE: {}".format(rmse))
        print("  MAE: {}".format(mae))
        print("  R2: {}".format(r2))


        # ====================================================
        # 7. Log Parameters
        # ====================================================

        mlflow.log_param(
            "alpha",
            alpha
        )

        mlflow.log_param(
            "l1_ratio",
            l1_ratio
        )


        # ====================================================
        # 8. Log Metrics
        # ====================================================

        mlflow.log_metric(
            "rmse",
            rmse
        )

        mlflow.log_metric(
            "mae",
            mae
        )

        mlflow.log_metric(
            "r2",
            r2
        )


        # ====================================================
        # 9. Create Model Signature
        # ====================================================

        predictions = lr.predict(train_x)

        signature = infer_signature(
            train_x,
            predictions
        )


        # ====================================================
        # 10. Log Model to MLflow
        # ====================================================

        mlflow.sklearn.log_model(
            lr,
            name="model",
            signature=signature
        )


        # ====================================================
        # 11. Display Run Information
        # ====================================================

        run_id = mlflow.active_run().info.run_id

        print("\nMLflow run completed successfully.")

        print(f"Run ID: {run_id}")

        print(
            "Tracking URI:",
            mlflow.get_tracking_uri()
        )
