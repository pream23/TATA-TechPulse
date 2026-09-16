import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def clean_dataframe(df):
    df=df.copy().drop_duplicates()
    numeric=df.select_dtypes(include=np.number).columns
    for c in numeric:
        q1,q3=df[c].quantile([.25,.75]); iqr=q3-q1
        lo,hi=q1-1.5*iqr,q3+1.5*iqr
        df[c]=df[c].clip(lo,hi)
    return df

def build_preprocessor(df):
    num=df.select_dtypes(include=np.number).columns
    cat=df.select_dtypes(exclude=np.number).columns
    return ColumnTransformer([
        ("num",Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler())]),num),
        ("cat",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),
                         ("onehot",OneHotEncoder(handle_unknown="ignore"))]),cat)
    ])

if __name__=="__main__":
    df=pd.DataFrame({"age":[21,22,np.nan,24,25,200],"income":[25000,30000,28000,np.nan,35000,40000],
                     "fuel":["petrol","diesel","petrol",None,"diesel","petrol"]})
    print("Before:",df)
    cleaned=clean_dataframe(df)
    X=build_preprocessor(cleaned).fit_transform(cleaned)
    print("After preprocessing shape:",X.shape)
