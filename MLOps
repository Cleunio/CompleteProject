mlflow.set_experiment("MLFlow Quickstart")

with mlflow.start_run():

    # log dos hiperparâmetros
    mlflow.log_params(params)

    # log da métrica
    mlflow.log_metric("accuracy", accuracy)

    # tag informativa
    mlflow.set_tag("Train info", "Logistic Regression - Iris data")

    # assinatura do modelo
    signature = infer_signature(X_Train, lr.predict(X_Train))

    # log do modelo 
    model_info = mlflow.sklearn.log_model(
        sk_model=lr,
        name="iris_model",                 # substitui artifact_path
        signature=signature,
        input_example=X_Train,
        registered_model_name="Tracking_quickstart"
    )
