import numpy as np
import kmeans
import common
import naive_em
import em

X = np.loadtxt("toy_data.txt")

# TODO: Your code here
K=4
seed=[0,1,2,3,4]
for s in seed:
    mixture,post=common.init(X,K,s)
    mixture, post, cost=kmeans.run(X,mixture,post)
    common.plot(X,mixture,post,'Plot in seed {}'.format(s))
    print(cost,mixture,post,s)