import numpy as np

def policy(w, state):
    x=np.tanh(state @ w["W1"] + w["b1"])
    return np.tanh(x @ w["W2"] + w["b2"])

def random_agent(inp=5, hidden=8, out=2, rng=None):
    rng=rng or np.random.default_rng()
    return {"W1":rng.normal(0,.5,(inp,hidden)),"b1":np.zeros(hidden),
            "W2":rng.normal(0,.5,(hidden,out)),"b2":np.zeros(out)}

def score(agent, rng, steps=80):
    pos, speed, lane, reward = 0., 0., 0., 0.
    for _ in range(steps):
        state=np.array([pos/steps,speed,lane,np.sin(pos*.1),1.])
        steer, throttle=policy(agent,state)
        lane += .08*steer
        speed=np.clip(speed+.12*throttle-.03,0,1)
        pos += speed
        reward += speed - .35*abs(lane)
    return reward

def mutate(a,rng,rate=.08):
    b={k:v.copy() for k,v in a.items()}
    for k in b: b[k]+=rng.normal(0,rate,b[k].shape)
    return b

def crossover(a,b,rng):
    c={}
    for k in a:
        mask=rng.random(a[k].shape)<.5
        c[k]=np.where(mask,a[k],b[k])
    return c

rng=np.random.default_rng(7)
pop=[random_agent(rng=rng) for _ in range(30)]
for gen in range(25):
    scores=np.array([score(a,rng) for a in pop])
    elite=[pop[i] for i in scores.argsort()[-6:]]
    new=elite.copy()
    while len(new)<len(pop):
        p1 = elite[rng.integers(0, len(elite))]
        p2 = elite[rng.integers(0, len(elite))]
        new.append(mutate(crossover(p1,p2,rng),rng))
    pop=new
    print(f"Generation {gen+1:02d} | best={scores.max():.2f} | mean={scores.mean():.2f}")
