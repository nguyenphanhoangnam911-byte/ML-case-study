import os
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"exp_common.py")).read())
res=[]
for sd in SEEDS:
    t=time.time(); r={"seed":sd}
    km=KMeans(K,init="k-means++",n_init=10,random_state=sd).fit(X)
    a=S.CustomKMeans(K,init="random",n_init=10,random_state=sd).fit(X)
    b=S.CustomKMeans(K,init="k-means++",n_init=10,random_state=sd).fit(X)
    w=lambda m:S.compute_inertia(X,m.labels_,m.cluster_centers_)
    r["km"]={"sk":w(km),"sc_rand":w(a),"sc_pp":w(b),"ari_rand":ARI(km.labels_,a.labels_),"ari_pp":ARI(km.labels_,b.labels_),
             "min_sk":int(np.bincount(km.labels_).min()),"min_rand":int(np.bincount(a.labels_).min()),"min_pp":int(np.bincount(b.labels_).min())}
    d=MiniBatchKMeans(K,batch_size=1024,n_init=10,random_state=sd).fit(X)
    runs=[MiniBatchKMeans(K,batch_size=1024,n_init=1,reassignment_ratio=0.0,random_state=sd+s).fit(X) for s in range(10)]
    f=min(runs,key=w)
    c=S.CustomMiniBatchKMeans(K,batch_size=1024,init="random",n_init=10,random_state=sd).fit(X)
    r["mb"]={"sk_default":w(d),"sk_fair":w(f),"sc":w(c),"ari_default":ARI(d.labels_,c.labels_),"ari_fair":ARI(f.labels_,c.labels_)}
    md=KMedoids(K,metric="euclidean",method="pam",init="k-medoids++",random_state=sd).fit(X)
    mc=S.CustomKMedoids(K,metric="euclidean",n_init=3,random_state=sd).fit(X)
    cost=lambda m:S.compute_abs_cost(X,m.labels_,m.cluster_centers_)
    r["kmed"]={"sk":cost(md),"sc":cost(mc),"ari":ARI(md.labels_,mc.labels_),
               "medoid_overlap":len(set(md.medoid_indices_.tolist())&set(mc.medoid_indices_.tolist()))}
    r["sec"]=time.time()-t; res.append(r); print("A seed",sd,round(r["sec"]),"s",flush=True)
    json.dump(res,open(os.path.join(HERE,"results","resA.json"),"w"),indent=1)
print("DONE_A")
