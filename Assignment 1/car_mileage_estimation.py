import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def demo_data(n=400, seed=42):
    rng=np.random.default_rng(seed)
    df=pd.DataFrame({
        "cylinders":rng.integers(4,9,n),
        "displacement":rng.uniform(70,455,n),
        "horsepower":rng.uniform(45,230,n),
        "weight":rng.uniform(1600,5000,n),
        "acceleration":rng.uniform(8,25,n),
        "model_year":rng.integers(70,83,n),
    })
    df["mpg"]=48 - .025*df.displacement - .004*df.weight - .025*df.horsepower + .35*df.acceleration + .45*(df.model_year-70)+rng.normal(0,2,n)
    return df

def main(path=None):
    df=pd.read_csv(path) if path else demo_data()
    aliases={"mpg":"mpg","MPG":"mpg","miles_per_gallon":"mpg"}
    df=df.rename(columns={c:aliases[c] for c in df.columns if c in aliases})
    target="mpg"
    if target not in df: raise ValueError("CSV must contain an mpg target column.")
    X=df.drop(columns=[target])
    y=pd.to_numeric(df[target],errors="coerce")
    numeric=X.select_dtypes(include=np.number).columns.tolist()
    prep=ColumnTransformer([("num",Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())]),numeric)],remainder="drop")
    model=Pipeline([("prep",prep),("rf",RandomForestRegressor(n_estimators=250,random_state=42))])
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42)
    model.fit(Xtr,ytr)
    pred=model.predict(Xte)
    print("MAE:",mean_absolute_error(yte,pred))
    print("RMSE:",mean_squared_error(yte,pred)**.5)
    print("R2:",r2_score(yte,pred))

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--data"); main(p.parse_args().data)
