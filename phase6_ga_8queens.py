
# PHASE 6: Genetic Algorithm for Eight Queens Problem

import random
import matplotlib.pyplot as plt

POP_SIZE = 100
GENS = 200
MUT_RATE = 0.1

def fitness(board):
    conflicts = 0
    for i in range(8):
        for j in range(i+1,8):
            if board[i] == board[j] or abs(board[i]-board[j]) == j-i:
                conflicts += 1
    return 28 - conflicts

def random_board():
    return [random.randint(0,7) for _ in range(8)]

def crossover(p1, p2):
    point = random.randint(1,7)
    return p1[:point] + p2[point:]

def mutate(board):
    if random.random() < MUT_RATE:
        board[random.randint(0,7)] = random.randint(0,7)
    return board

population = [random_board() for _ in range(POP_SIZE)]
best_scores = []

for _ in range(GENS):
    population = sorted(population, key=fitness, reverse=True)
    best_scores.append(fitness(population[0]))

    if fitness(population[0]) == 28:
        break

    new_pop = population[:10]
    while len(new_pop) < POP_SIZE:
        p1, p2 = random.sample(population[:50], 2)
        child = crossover(p1, p2)
        child = mutate(child)
        new_pop.append(child)

    population = new_pop

print("Best Solution:", population[0])
print("Fitness:", fitness(population[0]))

plt.plot(best_scores)
plt.title("GA Convergence")
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.show()
