
"""Cost-Aware ATM Cash Replenishment with DQN.
Runs in Google Colab or locally. If nn5_daily_dataset_without_missing_values.csv
is present, it uses it. Otherwise download/unzip the NN5 no-missing dataset from
Zenodo DOI 10.5281/zenodo.4656117 and point --data to the CSV file.
"""
import argparse, copy, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple

SEED=42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Transition=namedtuple('Transition','s a r ns done')

def load_nn5(path, series=None):
    df=pd.read_csv(path)
    # Monash NN5 converted CSVs normally contain multiple numeric series columns.
    numeric=df.select_dtypes(include=[np.number])
    if numeric.shape[1]==0:
        # attempt coercion for semicolon/other parsing edge cases
        df=pd.read_csv(path, sep=None, engine='python')
        numeric=df.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')
    if numeric.shape[1]==0: raise ValueError('No numeric NN5 series found in CSV.')
    col=series if series in numeric.columns else numeric.columns[0]
    x=numeric[col].dropna().astype(float).to_numpy()
    if len(x)<250: raise ValueError('Selected series is too short; expected a daily NN5 series.')
    # Scale demand from training data only later; raw values returned here.
    return x, str(col)

class ATMEnv:
    def __init__(self, demand, scale, capacity=8.0, quantity=6.0, horizon=56,
                 repl_cost=1.0, hold_cost=0.12, shortage_cost=8.0, lag=7):
        self.d=np.asarray(demand,dtype=np.float32)/scale
        self.capacity=float(capacity); self.quantity=float(quantity); self.horizon=horizon
        self.cr=repl_cost; self.ch=hold_cost; self.cs=shortage_cost; self.lag=lag
        self.rng=np.random.default_rng(SEED); self.reset()
    @property
    def state_dim(self): return 1+self.lag+2+1 # cash, history, sin/cos dow, days since replenishment
    def reset(self, start=None):
        lo=self.lag; hi=max(lo+1,len(self.d)-self.horizon)
        self.t=int(start if start is not None else self.rng.integers(lo,hi))
        self.end=min(len(self.d),self.t+self.horizon); self.cash=self.capacity; self.days_since=0
        self.metrics={'cost':0.0,'replenishments':0,'unmet':0.0,'cash_sum':0.0,'available_days':0,'days':0,'max_shortage_streak':0,'streak':0}
        return self._state()
    def _state(self):
        hist=self.d[self.t-self.lag:self.t]
        if len(hist)<self.lag: hist=np.pad(hist,(self.lag-len(hist),0))
        dow=self.t%7
        return np.array([self.cash/self.capacity,*np.clip(hist/self.capacity,0,2),np.sin(2*np.pi*dow/7),np.cos(2*np.pi*dow/7),min(self.days_since/14,1)],dtype=np.float32)
    def step(self,a):
        if a not in (0,1): raise ValueError('Action must be 0 or 1')
        before=self.cash
        added=min(self.quantity,self.capacity-before) if a==1 else 0.0
        available=before+added; demand=float(self.d[self.t]); unmet=max(0.0,demand-available)
        self.cash=max(0.0,available-demand)
        rc=self.cr*float(a); hc=self.ch*(self.cash/self.capacity); sc=self.cs*(unmet/max(1e-6,float(np.mean(self.d))))
        cost=rc+hc+sc; reward=-cost
        if a: self.days_since=0; self.metrics['replenishments']+=1
        else: self.days_since+=1
        self.metrics['cost']+=cost; self.metrics['unmet']+=unmet; self.metrics['cash_sum']+=self.cash
        self.metrics['days']+=1; ok=unmet<=1e-9; self.metrics['available_days']+=int(ok)
        self.metrics['streak']=0 if ok else self.metrics['streak']+1
        self.metrics['max_shortage_streak']=max(self.metrics['max_shortage_streak'],self.metrics['streak'])
        self.t+=1; done=self.t>=self.end
        ns=np.zeros(self.state_dim,dtype=np.float32) if done else self._state()
        return ns,reward,done,{'cost':cost,'unmet':unmet}
    def summary(self):
        m=self.metrics; n=max(1,m['days'])
        return {'total_cost':m['cost'],'replenishments':m['replenishments'],'cash_availability_pct':100*m['available_days']/n,
                'unmet_demand':m['unmet'],'avg_end_cash':m['cash_sum']/n,'max_consecutive_shortage_days':m['max_shortage_streak']}

class Net(nn.Module):
    def __init__(self,n):
        super().__init__(); self.m=nn.Sequential(nn.Linear(n,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,2))
    def forward(self,x): return self.m(x)

class Replay:
    def __init__(self,n=30000): self.b=deque(maxlen=n)
    def add(self,*x): self.b.append(Transition(*x))
    def sample(self,n): return Transition(*zip(*random.sample(self.b,n)))
    def __len__(self): return len(self.b)

def train(env, episodes=350, gamma=.99, lr=1e-3, batch=64, eps0=1.0, epsmin=.05, decay=.985):
    q=Net(env.state_dim).to(DEVICE); target=copy.deepcopy(q); opt=optim.Adam(q.parameters(),lr=lr); replay=Replay(); hist=[]; eps=eps0
    for ep in range(episodes):
        s=env.reset(); total=0.0
        while True:
            a=random.randrange(2) if random.random()<eps else int(q(torch.tensor(s,device=DEVICE).unsqueeze(0)).argmax(1).item())
            ns,r,done,_=env.step(a); replay.add(s,a,r,ns,done); s=ns; total+=r
            if len(replay)>=batch:
                z=replay.sample(batch); S=torch.tensor(np.array(z.s),dtype=torch.float32,device=DEVICE); A=torch.tensor(z.a,device=DEVICE).long().unsqueeze(1)
                R=torch.tensor(z.r,dtype=torch.float32,device=DEVICE); NS=torch.tensor(np.array(z.ns),dtype=torch.float32,device=DEVICE); D=torch.tensor(z.done,dtype=torch.float32,device=DEVICE)
                pred=q(S).gather(1,A).squeeze(1)
                with torch.no_grad(): y=R+gamma*(1-D)*target(NS).max(1).values
                loss=nn.functional.smooth_l1_loss(pred,y); opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(q.parameters(),10); opt.step()
            if done: break
        hist.append(total); eps=max(epsmin,eps*decay)
        if (ep+1)%10==0: target.load_state_dict(q.state_dict())
    return q,np.asarray(hist)

def run_policy(env, policy, starts):
    rows=[]
    for st in starts:
        s=env.reset(int(st))
        while True:
            a=policy(s,env)
            s,_,done,_=env.step(a)
            if done: break
        rows.append(env.summary())
    return pd.DataFrame(rows)

def main(args):
    raw,col=load_nn5(args.data,args.series)
    n=len(raw); i1=int(.70*n); i2=int(.85*n); scale=max(1e-6,float(np.percentile(raw[:i1],95)))
    train_d, val_d, test_d=raw[:i1],raw[i1-7:i2],raw[i2-7:]
    env=ATMEnv(train_d,scale); q,hist=train(env,args.episodes)
    testenv=ATMEnv(test_d,scale); maxstart=max(testenv.lag+1,len(testenv.d)-testenv.horizon)
    starts=np.unique(np.linspace(testenv.lag,maxstart-1,num=min(10,max(1,maxstart-testenv.lag)),dtype=int))
    def dqn(s,e):
        with torch.no_grad(): return int(q(torch.tensor(s,device=DEVICE).unsqueeze(0)).argmax(1).item())
    def fixed(s,e): return 1 if e.days_since>=6 else 0
    def threshold(s,e): return 1 if e.cash/e.capacity<0.35 else 0
    frames=[]
    for name,p in [('DQN',dqn),('Fixed schedule',fixed),('Cash threshold',threshold)]:
        x=run_policy(testenv,p,starts); x['policy']=name; frames.append(x)
    results=pd.concat(frames,ignore_index=True)
    summary=results.groupby('policy').agg(['mean','std']).round(4)
    torch.save({'state_dict':q.state_dict(),'state_dim':env.state_dim,'series':col,'scale':scale},args.model)
    pd.DataFrame({'episode':np.arange(1,len(hist)+1),'reward':hist}).to_csv(args.training_log,index=False)
    results.to_csv(args.results,index=False); summary.to_csv(args.summary)
    print('Series:',col,'days:',n,'device:',DEVICE); print('\nTEST SUMMARY\n',summary.to_string()); print('\nSaved model/log/results.')
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--data',default='nn5_daily_dataset_without_missing_values.csv'); p.add_argument('--series',default=None); p.add_argument('--episodes',type=int,default=350)
    p.add_argument('--model',default='atm_dqn.pt'); p.add_argument('--training-log',default='training_log.csv'); p.add_argument('--results',default='test_results.csv'); p.add_argument('--summary',default='test_summary.csv'); main(p.parse_args())

