"""Bradley--Terry fitting for within-task trajectory comparisons."""
import math

def bradley_terry(episodes, matches, iters=500, prior=0.5):
    """MM algorithm with a weak prior (each episode gets `prior` virtual wins/losses vs the mean)."""
    idx = {e: i for i, e in enumerate(episodes)}
    n = len(episodes)
    wins = [[0.0] * n for _ in range(n)]
    for (a, b), per in matches.items():
        if a not in idx or b not in idx:
            continue
        for lab in per.values():
            if lab == a:
                wins[idx[a]][idx[b]] += 1
            elif lab == b:
                wins[idx[b]][idx[a]] += 1
            else:
                wins[idx[a]][idx[b]] += 0.5; wins[idx[b]][idx[a]] += 0.5
    p = [1.0] * n
    for _ in range(iters):
        newp = []
        for i in range(n):
            W = sum(wins[i]) + prior
            denom = prior / (p[i] + 1.0)
            for j in range(n):
                if i == j:
                    continue
                nij = wins[i][j] + wins[j][i]
                if nij:
                    denom += nij / (p[i] + p[j])
            newp.append(W / denom if denom > 0 else p[i])
        s = sum(newp) / n
        p = [x / s for x in newp]
    return {e: math.log(p[idx[e]]) for e in episodes}
