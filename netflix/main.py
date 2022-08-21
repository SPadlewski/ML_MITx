import numpy as np
import kmeans
import common
import naive_em
import em

X = np.loadtxt("toy_data.txt")

# TODO: Your code here
K=[1,2,3,4]
seed=[0,1,2,3,4]
for k in K:
    cost_list=[]
    post_list=[]
    mixture_list=[]
    for s in seed:
        mixture,post=common.init(X,k,s)
        mixture, post, cost=kmeans.run(X,mixture,post)
        cost_list.append(cost)
        post_list.append(post)
        mixture_list.append(mixture)
    min_cost_value=min(cost_list)
    min_index=cost_list.index(min_cost_value)
    common.plot(X,mixture_list[min_index],post_list[min_index],'Plot in seed {}'.format(s))
    print(cost_list[min_index],k)