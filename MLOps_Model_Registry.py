model_name = "tracking-quickstart"
model_version="lastest"

model_uri = f"models:/{model_name}/{model_version}
model=mlflow.sklearn.load_model(model_uri)
model


y_pred_new = model.predict(X_test)
y_pred_new
