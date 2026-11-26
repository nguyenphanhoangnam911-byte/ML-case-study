"""Hình 11 (một Episode của MC Control). Chạy: MC_DATA=/đường/dẫn/hotel_bookings.csv python make_figP9.py"""
import sys,numpy as np
import os
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,os.path.join(HERE,'src'))
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mc_pricing import *
df,c=load_bookings(os.environ.get('MC_DATA',os.path.join(HERE,'notebooks','hotel_bookings.csv')))
obs,p_hist,n=booking_curves(c); print(n)
mdp=PricingMDP(Scenario(obs,p_hist)); V,Q,pi=mdp.solve(); print(V[30,10])
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
P=mdp.prices.astype(int); PC=['#9ecae1','#4a90c9','#3a9a5c','#e0a030','#b0413e']
L=OnPolicyMC(mdp,seed=11,eps_schedule=lambda k:0.2)
for _ in range(150): L.run_episode()
import copy
Qb=copy.deepcopy(L.Q); Nb=copy.deepcopy(L.N); gb=copy.deepcopy(L.greedy)
# one more episode, record
L.k+=1; ep=L.sim.rollout(mdp.T,mdp.C,lambda t,x:L.eps_act(t,x,0.2))
G=0.;Gs=[]
for t,x,a,r in reversed(ep): G+=r; Gs.append(G)
Gs=Gs[::-1]; print(len(ep),ep[:4],Gs[:3])
L.revenue.append(Gs[0]); update_mc(L,ep)
fig,ax=plt.subplots(2,2,figsize=(11,8.4))
a=ax[0,0]; ts=[e[0] for e in ep]; xs=[e[1] for e in ep]
a.step(ts,xs,where='post',color='#777',lw=1.2)
for t,x,aa,r in ep: a.scatter(t,x,s=70,color=PC[aa],edgecolor='k',zorder=3)
a.invert_xaxis(); a.set_xlabel('Số ngày còn lại t'); a.set_ylabel('Số phòng còn lại x'); a.set_title('(a) Một Episode: trạng thái (t, x) và mức giá được chọn',fontsize=10,loc='left')
a.legend(handles=[plt.Line2D([],[],marker='o',ls='',color=PC[i],mec='k',label=f'{P[i]} €') for i in range(5)],fontsize=8,ncol=5,loc='upper center',bbox_to_anchor=(.5,-.18),title='Giá',title_fontsize=8,frameon=False)
b=ax[0,1]; idx=np.arange(len(ep)); R=[e[3] for e in ep]
b.bar(idx-.2,R,.4,color='#7aa6d6',label='Phần thưởng tức thời r_t'); b.bar(idx+.2,Gs,.4,color='#d9822b',label='Return G_t (cộng ngược từ cuối)')
b.set_xticks(idx); b.set_xticklabels([f't={t}' for t in ts],rotation=90,fontsize=7); b.set_ylabel('€'); b.set_ylim(0,820); b.legend(fontsize=8,frameon=False,ncol=2,loc='upper center'); b.set_title('(b) Return G_t tính ngược từ cuối Episode (γ = 1)',fontsize=10,loc='left')
c=ax[1,0]; w=.27
qb=[Qb[t][x][a_] for t,x,a_,_ in ep]; qa=[L.Q[t][x][a_] for t,x,a_,_ in ep]
c.bar(idx-w,qb,w,color='#bbb',label='Q trước'); c.bar(idx,Gs,w,color='#d9822b',label='Mẫu G_t'); c.bar(idx+w,qa,w,color='#2a6fb0',label='Q sau = Q + (G − Q)/N')
c.set_xticks(idx); c.set_xticklabels([f'({t},{x})' for t,x,_,_ in ep],rotation=90,fontsize=7); c.set_ylabel('€'); c.set_ylim(0,1150); c.legend(fontsize=8,frameon=False,ncol=3,loc='upper center'); c.set_title('(c) Cập nhật Q(t, x, a) tại các ô vừa đi qua (sau 150 Episode)',fontsize=10,loc='left')
d=ax[1,1]; t0,x0,a0,_=ep[0]; w=.27; ii=np.arange(5)
d.bar(ii-w,Qb[t0][x0],w,color='#bbb',label='Q trước'); d.bar(ii,L.Q[t0][x0],w,color='#2a6fb0',label='Q sau Episode'); d.bar(ii+w,Q[t0,x0],w,color='#3a9a5c',label='Q* (quy hoạch động)')
d.set_xticks(ii); d.set_xticklabels([f'{p} €' for p in P]); d.set_ylabel('€'); d.set_ylim(850,1075); d.legend(fontsize=8,frameon=False,ncol=3,loc='upper center')
gn=int(np.argmax(L.Q[t0][x0])); d.set_title(f'(d) Q tại ({t0},{x0}): tham lam sau cập nhật = {P[gn]} €,\n      tối ưu (DP) = {P[int(pi[t0,x0])]} € (trục y bắt đầu từ 850)',fontsize=10,loc='left')
fig.suptitle('Từng bước của Monte Carlo Control (on-policy, ε = 0,2, giá 85–150 €, T = 30, C = 10) trên dữ liệu đặt phòng khách sạn',fontsize=11,x=.01,ha='left')
fig.tight_layout(rect=(0,0,1,.97)); fig.savefig(os.path.join(HERE,'results','figP9_algorithm_steps.png'),dpi=160)
print(Qb[t0][x0],L.Q[t0][x0],Q[t0,x0],a0)
