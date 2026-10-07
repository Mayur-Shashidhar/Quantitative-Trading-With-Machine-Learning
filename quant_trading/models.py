from dataclasses import dataclass
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

def model_templates(seed=42):
    return {"Elastic Net": ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10000, random_state=seed), "Decision Tree": DecisionTreeRegressor(max_depth=5, min_samples_leaf=20, random_state=seed), "XGBoost": XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, objective="reg:squarederror", random_state=seed), "LightGBM": LGBMRegressor(n_estimators=300, learning_rate=0.03, max_depth=4, num_leaves=20, subsample=0.8, colsample_bytree=0.8, verbosity=-1, random_state=seed)}

@dataclass
class FittedModel:
    model: object
    scaler: StandardScaler | None = None
    def predict(self, features: pd.DataFrame):
        return self.model.predict(self.scaler.transform(features) if self.scaler is not None else features)

def fit_model(name, template, features, target):
    scaler = StandardScaler() if name == "Elastic Net" else None; values = scaler.fit_transform(features) if scaler is not None else features
    model = clone(template); model.fit(values, target); return FittedModel(model, scaler)
