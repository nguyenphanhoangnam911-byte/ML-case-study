import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
FEATURES=["Recency","Frequency","Monetary"]
NON={"POST","DOT","C2","M","S","B","D","BANK CHARGES","AMAZONFEE","CRUK","PADS"}
def load(path):
    raw=pd.read_csv(path,encoding="latin1")
    raw["InvoiceDate"]=pd.to_datetime(raw["InvoiceDate"],format="%m/%d/%Y %H:%M")
    df=raw[raw["CustomerID"].notna()]; df=df[df["UnitPrice"]>0]
    df=df[~df["StockCode"].astype(str).isin(NON)].copy()
    df["CustomerID"]=df["CustomerID"].astype(int)
    df["TotalPrice"]=df["Quantity"]*df["UnitPrice"]
    df["IsCancel"]=df["InvoiceNo"].astype(str).str.startswith("C")
    df=df.reset_index(drop=True)
    pur=df[~df["IsCancel"]]
    snap=pur["InvoiceDate"].max()+pd.Timedelta(days=1)
    a=pur.groupby("CustomerID").agg(LastPurchase=("InvoiceDate","max"),Frequency=("InvoiceNo","nunique"))
    a["Monetary"]=df.groupby("CustomerID")["TotalPrice"].sum().round(2)
    a["Recency"]=(snap-a["LastPurchase"]).dt.days
    rfm=a.loc[a["Monetary"]>0,FEATURES].copy()
    X=StandardScaler().fit_transform(rfm.to_numpy(dtype=float))
    rl=rfm.astype(float).copy(); rl[["Frequency","Monetary"]]=np.log1p(rl[["Frequency","Monetary"]])
    Xl=StandardScaler().fit_transform(rl.to_numpy())
    return rfm,X,Xl
