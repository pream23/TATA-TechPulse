import argparse, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

def data(n=500):
    r=np.random.default_rng(1)
    d=pd.DataFrame({"age":r.integers(0,15,n),"km":r.integers(5000,180000,n),
                    "engine":r.uniform(1,4,n),"brand":r.choice(["A","B","C"],n)})
    d["price"]=30000-1200*d.age-.07*d.km+3500*d.engine+(d.brand=="C")*2500+r.normal(0,1500,n)
    return d

def main(path=None):
    d=pd.read_csv(path) if path else data()
    y=d.pop("price")
    num=d.select_dtypes(include=np.number).columns
    cat=d.select_dtypes(exclude=np.number).columns
    prep=ColumnTransformer([("num",SimpleImputer(strategy="median"),num),
                            ("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),
                                             ("oh",OneHotEncoder(handle_unknown="ignore"))]),cat)])
    models={"Linear":LinearRegression(),"Ridge":Ridge(alpha=10),"RandomForest":RandomForestRegressor(n_estimators=250,random_state=42)}
    Xt,Xv,yt,yv=train_test_split(d,y,test_size=.2,random_state=42)
    for name,m in models.items():
        pipe=Pipeline([("prep",prep),("model",m)]); pipe.fit(Xt,yt); p=pipe.predict(Xv)
        print(name, "MAE=",round(mean_absolute_error(yv,p),2),"R2=",round(r2_score(yv,p),3))

if __name__=="__main__":
    a=argparse.ArgumentParser();a.add_argument("--data");main(a.parse_args().data)
