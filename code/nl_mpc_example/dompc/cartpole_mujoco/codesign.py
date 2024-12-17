import numpy as np
from main import balance

# do control co-design to optimize the length of the 
# pole in cart pole system to minimize force used to balance

cost = 25000
cost_prev = 0
step = 0.1
thres = 5000
psize_cur = 0.3
psize_prev = 0

iter = 1
while (cost > thres):

    if iter != 1: # ignore if first iteration
        diff_cost = cost - cost_prev
        diff_size = psize_cur - psize_prev
        psize_prev = psize_cur
        if diff_cost > 0: # if cost went up with last step
            if diff_size > 0: # and if pole size increased
                psize_cur =- step # decrease pole size for next step
            else: # and if pole size decreased
                psize_cur =+ step #increase pole size for next step
        else: # if cost went down
            if diff_size > 0: # and if pole size increased
                psize_cur =+ step # decrease pole size for next step
            else: # and if pole size decreased
                psize_cur =- step #increase pole size for next step
    
    cost_prev = cost

    cost, tf = balance(polelen = psize_cur)
    print(cost, psize_cur, iter)
    iter += 1
    

print(f'Reached Threshhold Cost of {cost} with pole size of {psize_cur}')

        


    
