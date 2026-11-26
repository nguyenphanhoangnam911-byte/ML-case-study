import sys,json,time
import os
HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,HERE); sys.path.insert(0,os.path.join(HERE,"..","src"))
import numpy as np, warnings
warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn_extra.cluster import KMedoids
import rfm_scratch as S
from rfm_prep import load
rfm,X,Xl=load(os.environ.get("RFM_DATA", os.path.join(HERE,"..","notebooks","data.csv")))
K=4; SEEDS=[0,1,2,3,4]
