import numpy as np

opening_move_counts = np.array([
    74422566,
    39617077,
    7994832,
    6491388,
    2549067,
    2419609,
    2081360,
    1619141,
    1194039,
    967218,
    609295,
    398388,
])

opening_move_proportions = opening_move_counts / sum(opening_move_counts)

print('Top k accuracies:')
print(np.cumsum(opening_move_proportions))

print('Entropy:')
print(sum(-opening_move_proportions * np.log(opening_move_proportions) ))