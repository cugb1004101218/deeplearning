import os
import random

r_matrix = []
for i in range(0, 6):
    r_matrix.append([-1]*6)
r_matrix[0][4] = 0
r_matrix[4][0] = 0
r_matrix[4][3] = 0
r_matrix[4][5] = 100
r_matrix[1][5] = 100
r_matrix[3][4] = 0
r_matrix[3][2] = 0
r_matrix[3][1] = 0
r_matrix[2][3] = 0
r_matrix[1][5] = 100
r_matrix[1][3] = 0
r_matrix[5][1] = 0
r_matrix[5][4] = 0
r_matrix[5][5] = 100
print("r matrix:")
for i in range(0, 6):
    r_matrix.append([-1]*6)
    print(r_matrix[i])
q_matrix = []
print("q matrix:")
for i in range(0, 6):
    q_matrix.append([0]*6)
    print(q_matrix[i])

def q_trans(s, a):
    max_q = 0
    for i in range(0, 6):
        if r_matrix[a][i] == -1:
            continue
        if q_matrix[a][i] > max_q:
            max_q = q_matrix[a][i]
    q_matrix[s][a] = r_matrix[s][a] + 0.8 * max_q

def episode(s):
    while True:
        next_s = random.randint(0, 5)
        if r_matrix[s][next_s] == -1:
            continue
        q_trans(s, next_s)
        s = next_s
        if s == 5:
            return

if __name__ == '__main__':
    for i in range(0, 100000):
        start_s = random.randint(0, 5)
        episode(start_s)

    print("final q_matrix:")
    for i in range(0, 6):
        q_matrix.append([0]*6)
        print(q_matrix[i])
