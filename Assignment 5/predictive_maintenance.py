import argparse, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def make_data(n=1200):
    r=np.random.default_rng(3)
    d=pd.DataFrame({"temperature":r.normal(70,10,n),"torque":r.normal(55,15,n),
                    "speed":r.normal(1500,250,n),"tool_wear":r.uniform(0,250,n)})
    risk=.02*(d.temperature-70)+.012*(d.torque-55)+.008*(d.tool_wear-120)
    p=1/(1+np.exp(-risk))
    d["machine_failure"]=(r.random(n)<p*.18).astype(int)
    return d

def main(path=None):
    d=pd.read_csv(path) if path else make_data()
    target="machine_failure"
    X=d.drop(columns=[target]); y=d[target]
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,stratify=y,random_state=42)
    m=RandomForestClassifier(n_estimators=300,class_weight="balanced",random_state=42)
    m.fit(Xtr,ytr); pred=m.predict(Xte); proba=m.predict_proba(Xte)[:,1]
    print(classification_report(yte,pred,zero_division=0))
    print("ROC-AUC:",roc_auc_score(yte,proba))
    print("Confusion matrix:",confusion_matrix(yte,pred))

if __name__=="__main__":
    a=argparse.ArgumentParser();a.add_argument("--data");main(a.parse_args().data)
