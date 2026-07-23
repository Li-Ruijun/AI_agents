import numpy as np

def accuracy(memory,gamma,alpha):
    
    # Accuracy function.
    a_M = 1 - 0.5* np.exp(- gamma * (memory ** alpha))

    return a_M


# Calculate V_H
def calculate_remaining_honest_value(t, T, solo_accuracy, discount_factor):
    
    period = t + 1
    remaining_period = max(T - period, 0)
    discount_sum = sum( discount_factor ** tau for tau in range(1, remaining_period + 1))
    remaining_honest_value = solo_accuracy * discount_sum

    return remaining_honest_value



def calculate_belief_cutoff(
        t,
        T,
        beta,
        gamma,
        alpha,
        memory_endowment,
        collaboration_cost,
        discount_factor
):
    
    # Solo accuracy a_s = a(m)
    solo_accuracy = accuracy(
        memory=memory_endowment,
        gamma=gamma,
        alpha=alpha
    )

    # Pair accuracy a_p = a(2m)
    pair_accuracy = accuracy(
        memory= 2 * memory_endowment,
        gamma=gamma,
        alpha=alpha
    )

    # Collaboration benefit from pooling: g = a_p - a_s
    pooling_gain = pair_accuracy - solo_accuracy

    # Net gain from collaboration: g - c
    net_gain = pooling_gain - collaboration_cost

    # If collaboration has no positive net gain, do not collabotate
    if net_gain <= 0:
        return 0.0
    
    # If bate is 0, there is no contagion risk.
    if beta <= 0:
        return 1.0
    
    remaining_honest_value = calculate_remaining_honest_value(
        t=t,
        T=T,
        solo_accuracy=solo_accuracy,
        discount_factor=discount_factor
    )

    cutoff = net_gain / (beta * ((2 * pair_accuracy - 1) + discount_factor * remaining_honest_value))

    return cutoff
