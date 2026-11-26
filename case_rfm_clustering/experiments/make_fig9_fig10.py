"""Hình 9 (K-Means và PAM từng bước) và Hình 10 (ARI trước/sau khi bỏ outlier).
Chạy: RFM_DATA=/đường/dẫn/data.csv python experiments/make_fig9_fig10.py"""
import sys,json,numpy as np
import os
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.join(HERE,'..')
sys.path.insert(0,os.path.join(ROOT,'src')); sys.path.insert(0,HERE)
OUT=os.path.join(ROOT,'results')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from rfm_prep import load
from rfm_scratch import CustomKMeans, CustomKMedoids
rfm,X,Xl=load(os.environ.get('RFM_DATA',os.path.join(ROOT,'notebooks','data.csv'))); print(len(rfm), X.shape)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
C=['#2a6fb0','#d9822b','#3a9a5c','#b0413e']
M=Xl; K=4; rs=np.random.default_rng(7)
# --- K-Means manual run (random init), on X_log, plane Recency(0) vs Monetary(2)
km=CustomKMeans(K,init='random',n_init=1,random_state=7)
rng=np.random.default_rng(7); cen=km._init_centroids(M,rng); path=[cen.copy()]; hist=[]
for it in range(60):
    lab,d2=km._assign_clusters(M,cen); new=km._update_centroids(M,lab,d2); hist.append(float(d2.sum()))
    path.append(new.copy()); 
    if np.allclose(new,cen): break
    cen=new
path=np.array(path); n_it=len(hist); print('kmeans iters',n_it,hist[:3],hist[-1])
# --- PAM manual run
pam=CustomKMedoids(K,n_init=1,random_state=7); D=pam._distance_matrix(M)
med=pam._init_medoids(M.shape[0],np.random.default_rng(7)); _,cost=pam._assign_clusters(D,med)
ph=[cost]; mpath=[med.copy()]
for _ in range(100):
    nc,sl,cd=pam._best_swap(D,med,cost)
    if sl<0: break
    med[sl]=cd; cost=nc; ph.append(cost); mpath.append(med.copy())
mpath=np.array(mpath); print('pam swaps',len(ph)-1,ph)
lab_km,_=km._assign_clusters(M,path[-1]); lab_pam,_=pam._assign_clusters(D,med)
ix,iy=0,2
def scat(ax,lab,title):
    for k in range(K): ax.scatter(M[lab==k,ix],M[lab==k,iy],s=4,alpha=.35,color=C[k],lw=0)
    ax.set_xlabel('Recency (log-chuẩn hóa)' if False else 'Recency (chuẩn hóa)'); ax.set_ylabel('Monetary (log1p, chuẩn hóa)'); ax.set_title(title,fontsize=10,loc='left')
fig,ax=plt.subplots(2,2,figsize=(11,8.4))
a=ax[0,0]; lab0,_=km._assign_clusters(M,path[0]); scat(a,lab_km,f'(a) K-Means: quỹ đạo 4 tâm qua {n_it} vòng lặp')
for k in range(K):
    a.plot(path[:,k,ix],path[:,k,iy],'-',color='k',lw=1,alpha=.8)
    a.scatter(path[:,k,ix],path[:,k,iy],s=14,color='k',zorder=3)
    a.scatter(*path[0,k,[ix,iy]],marker='s',s=70,facecolor='white',edgecolor='k',zorder=4)
    a.scatter(*path[-1,k,[ix,iy]],marker='*',s=240,facecolor=C[k],edgecolor='k',zorder=5)
a.text(.02,.03,'ô vuông: tâm khởi tạo · sao: tâm cuối',transform=a.transAxes,fontsize=8)
b=ax[0,1]; b.plot(range(1,n_it+1),hist,'o-',color=C[0]); b.set_xlabel('Vòng lặp'); b.set_ylabel('WCSS J'); b.set_title('(b) WCSS giảm đơn điệu sau mỗi vòng',fontsize=10,loc='left')
b.annotate(f'{hist[0]:.0f}',(1,hist[0]),textcoords='offset points',xytext=(6,0),fontsize=8); b.annotate(f'{hist[-1]:.0f}',(n_it,hist[-1]),textcoords='offset points',xytext=(-6,8),fontsize=8,ha='right')
c=ax[1,0]; scat(c,lab_pam,f'(c) PAM: medoid ban đầu → sau {len(ph)-1} lần hoán đổi')
for k in range(K):
    c.scatter(*M[mpath[0,k],[ix,iy]],marker='s',s=70,facecolor='white',edgecolor='k',zorder=4)
    c.scatter(*M[med[k],[ix,iy]],marker='*',s=240,facecolor=C[k],edgecolor='k',zorder=5)
for s in range(1,len(mpath)):
    sl=np.flatnonzero(mpath[s]!=mpath[s-1])[0]
    c.annotate('',xy=M[mpath[s,sl],[ix,iy]],xytext=M[mpath[s-1,sl],[ix,iy]],arrowprops=dict(arrowstyle='->',color='k',lw=.9,alpha=.7))
c.text(.02,.03,'ô vuông: medoid khởi tạo · sao: medoid cuối · mũi tên: hoán đổi giữ lại',transform=c.transAxes,fontsize=8)
d=ax[1,1]; d.step(range(len(ph)),ph,where='post',marker='o',color=C[1]); d.set_xlabel('Lần hoán đổi được giữ'); d.set_ylabel('Tổng chi phí tuyệt đối T'); d.set_title('(d) Chi phí T giảm sau mỗi hoán đổi tốt nhất',fontsize=10,loc='left')
fig.suptitle('Từng bước của K-Means và K-Medoids (PAM) trên 4.320 khách hàng (K = 4, seed 7, không gian X_log)',fontsize=11,x=.01,ha='left')
fig.tight_layout(rect=(0,0,1,.97)); fig.savefig(os.path.join(OUT,'fig9_algorithm_steps.png'),dpi=160); plt.close(fig)
# --- outlier before/after
R=json.load(open(os.path.join(HERE,'results','resB.json')))
models=['K-Means','Mini-Batch K-Means','K-Medoids (PAM)']; fig,ax=plt.subplots(1,2,figsize=(10.5,4.2),sharey=True)
for a,sp,t in zip(ax,['X','X_log'],['Không gian X (chuẩn hóa)','Không gian X_log (log1p + chuẩn hóa)']):
    for i,m in enumerate(models):
        v=[r['ari_core'] for r in R['rows'] if r['space']==sp and r['model']==m]
        a.bar(i,np.mean(v),yerr=np.std(v),color=C[i],capsize=4,width=.6); a.scatter([i]*len(v),v,color='k',s=12,zorder=3)
        a.text(i,.03,f'{np.mean(v):.2f}',ha='center',color='w',fontsize=10,fontweight='bold')
    a.set_xticks(range(3)); a.set_xticklabels(['K-Means','Mini-Batch','K-Medoids'],fontsize=9); a.set_title(t,fontsize=10,loc='left'); a.set_ylim(0,1.05)
ax[0].set_ylabel('ARI giữa nhãn trước và sau khi bỏ outlier')
fig.suptitle(f'Mức thay đổi nhãn khi bỏ {R["n_out"]} outlier (|z|>3): ARI trung bình ± độ lệch chuẩn, 5 seed (chấm: từng seed)',fontsize=10.5,x=.01,ha='left')
fig.tight_layout(rect=(0,0,1,.93)); fig.savefig(os.path.join(OUT,'fig10_outlier_removal.png'),dpi=160)
for sp in ['X','X_log']:
    print(sp,{m:round(float(np.mean([r['ari_core'] for r in R['rows'] if r['space']==sp and r['model']==m])),3) for m in models})
