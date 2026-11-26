import os
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"exp_common.py")).read())
is_out=(np.abs(X)>3).any(axis=1); core=~is_out
def mk(name,sd):
    if name=="K-Means": return KMeans(K,n_init=10,random_state=sd)
    if name=="Mini-Batch K-Means": return MiniBatchKMeans(K,batch_size=1024,n_init=10,random_state=sd)
    return KMedoids(K,metric="euclidean",method="pam",init="k-medoids++",random_state=sd)
res={"n_out":int(is_out.sum()),"rows":[]}
for sd in SEEDS:
    t=time.time()
    for sp,M in (("X",X),("X_log",Xl)):
        for name in ("K-Means","Mini-Batch K-Means","K-Medoids (PAM)"):
            full=mk(name,sd).fit(M).labels_
            lc=mk(name,sd).fit_predict(M[core]); la=mk(name,sd+1000).fit_predict(M[core])
            sizes=np.bincount(full,minlength=K)
            share=np.bincount(full,weights=is_out.astype(float),minlength=K)/np.maximum(sizes,1)
            res["rows"].append({"seed":sd,"space":sp,"model":name,"ari_core":ARI(full[core],lc),"ari_base":ARI(lc,la),
                                "n_out_clusters":int((share>0.5).sum()),"min_size":int(sizes.min())})
    print("B seed",sd,round(time.time()-t),"s",flush=True)
    json.dump(res,open(os.path.join(HERE,"results","resB.json"),"w"),indent=1)
print("DONE_B")
