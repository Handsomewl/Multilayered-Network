import numpy as np
from MinTree import MinTree
from scipy.sparse import csc_matrix, coo_matrix, csr_matrix, lil_matrix


# Greedy search for find a dense subgraph i.e. clique in a monopartite graph.
# Input : sparse graph adjacency matrix
# Output : subgraph and its corresponding score
def fastGreedyDecreasing(G):
    # Mcur is a sysmmetric matrix.
    # Mcur : lil_matrix
    Mcur = G.tolil()
    curScore = Mcur.sum() / 2
    curSet = set(range(0, Mcur.shape[1]))
    bestAveScore = 2 * curScore / (len(curSet)*(len(curSet)-1))
    Deltas = np.squeeze(Mcur.sum(axis=1).A)
    tree = MinTree(Deltas)

    numDeleted = 0
    deleted = []
    bestNumDeleted = 0

    # Constaint to avoid trivial solution in finding a dense clique.
    while len(curSet) > 2:
        node, val = tree.getMin()
        curScore -= val
        # Update priority for the node with min priority and its neighbors.
        for j in Mcur.rows[node]:
            delt = Mcur[node, j]
            tree.changeVal(j, -delt)
        curSet -= {node}
        tree.changeVal(node, float('inf'))
        deleted.append(node)
        numDeleted += 1
        curAveScore = 2 * curScore / (len(curSet)*(len(curSet)-1))
        if curAveScore > bestAveScore:
            bestAveScore = curAveScore
            bestNumDeleted = numDeleted

    # reconstruct the best sets
    finalSet = set(range(0, Mcur.shape[1]))
    for idx in range(bestNumDeleted):
        finalSet.remove(deleted[idx])
    return finalSet, bestAveScore
