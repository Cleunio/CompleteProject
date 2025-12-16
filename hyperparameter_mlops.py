## Hyperparameter tuning using Grid Searchcv
def hyperparameter_tuning(X_train, y_train, param_grid):
  rf=RandomForestRegressor()
  grid_search=GridSearchCV(estimator=rf, param_grid=param_grid,cv=3,n_jobs=-1, verbose=2,scoring="neg_mean_squared_error")
  grid_search.fit(X_train, y_train)
  return grid_search

#Train Test Split, Hyperparameter Tuning Mlflow Experiments

from urllib.parse import urlparse
#Independent and Dependent Features
X = data.drop(columns=["Price"])
y=data["Price"]

##Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20)
from mlflow.models import infer_signature
signature=infer_signature(X_train, y_train)

param_grid = {
  'n_estimators':[100, 200],
  'max_depth': [5, 10, None],
  'min_samples_split': [2, 5,],
  'min_samples_leaf': [1, 2]
}
##start the Mlflow Experiments
with mlflow.start_run():
  ##Perform hyperparameter_tuning
  grid_search = hyperparameter_tuning(X_train, y_train, param_grid)
  ## Get the best model
  best_model=grid_search.best_estimator_
  ##Evaluate the best model
  y_pred=best_model.predict(X_test)
  mse = mean_squared_error(y_test, y_pred)
  ## log best parameters
  mlflow.log_param("best_n_estimators", grid_search.best_params_['n_estimators'])
  mlflow.log_param("best_max_depth", grid_search.best_params_['max_depth'])
  mlflow.log_param("best_min_samples_split", grid_search.best_params_['min_samples_split'])
  mlflow.log_param("best_min_samples_leaf", grid_search.best_params_['min_samples_leaf'])
  mlflow.log_metric("mse", mse)

  ## tracking url
  mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
  tracking_url_type_store=urlparse(mlflow.get_tracking_uri())

  if tracking_url_type_store != 'file':
    mlflow.sklearn.log_model(best_model,"model", registered_model_name="Best Randomforest Model")
  else:
    mlflow.sklearn.log_model(best_model, "model", signature=signature)
    
  print(f"Best Hyperparameter:{grid_search.best_paramS_}")
  print(f"Mean Squared Errror:{mse}")

  
