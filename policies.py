import numpy as np
from collections import Counter
from statsmodels.stats.proportion import proportion_confint
from scipy.stats import beta
from utils import sample_from_scores
from utils import pop_est
from utils import discount


class AbstractPolicy(object):
    
    def __init__(self, n, seed):
        self.n = n 
        self.t = 0
        self.seed = seed
        
    def select(self, regions, budget):
        raise NotImplementedError

    def estimate_pop(self, regions, budget):
        raise NotImplementedError

    @property
    def name(self):
        raise NotImplementedError

    @property
    def formal_name(self):
        raise NotImplementedError


class UCB(AbstractPolicy):

    def __init__(self, n, gamma=1, alpha=0.16, seed=0):
        super().__init__(n, seed)
        #self.gap = gap
        self.gamma = gamma
        self.alpha = alpha
        self.scores = None

    def select(self, regions, budget):

        if self.t == 0:
            # Sample everything at least once because we don't yet have enough
            # information to compute ucb.
            samples = np.arange(budget) % len(regions)
            self.scores = samples
            self.t += 1
            return Counter(samples)

        self.t += 1

        # Get ucb scores for each region
        scores = []
        #gp = min(self.gap, len(regions[0][self.name]['cases_seen']))
        for i in range(len(regions)):
            if discount(regions[i][self.name]['n_tested'][::-1], self.gamma) == 0:
                ucb = 1.0
            # if np.sum(regions[i][self.name]['n_tested'][-gp:]) == 0:
            #     ucb = 1.0
            else:
                ucb = proportion_confint(
                    discount(regions[i][self.name]['cases_seen'][::-1], self.gamma),
                    discount(regions[i][self.name]['n_tested'][::-1], self.gamma),
                    # np.sum(regions[i][self.name]['cases_seen'][-gp:]),
                    # np.sum(regions[i][self.name]['n_tested'][-gp:]),
                    alpha=self.alpha, method='beta'
                )[1]
            scores.append(ucb)

        self.scores = scores # record for population estimation

        return sample_from_scores(scores, budget, self.seed)

    def estimate_pop(self, regions, budget):

        pdf = self.scores / np.sum(self.scores)
        return pop_est(regions, pdf, self.name, m=budget)

    @property
    def name(self):
        return f'UCB_{self.alpha}'

    @property
    def formal_name(self):
        return f'UCB, $\\alpha={self.alpha}$'


class Egreedy(AbstractPolicy):

    def __init__(self, n, eps=0.1, gamma=1, seed=0):
        super().__init__(n, seed)
        self.eps = eps
        self.gamma = gamma

    def select(self, regions, budget):

        pr = lambda x, y: 0 if y == 0 else x / y
        i = np.argmax([
            pr(
                discount(regions[i][self.name]['cases_seen'][::-1], self.gamma),
                discount(regions[i][self.name]['n_tested'][::-1], self.gamma)
            )
            for i in range(len(regions))
        ])
        n_rand = int(budget * self.eps)

        exploit = Counter({i: budget - n_rand})
        np.random.seed(self.seed)
        explore = Counter(np.random.choice(len(regions), size=n_rand, replace=True))

        return exploit + explore

    @property
    def name(self):
        return f'egreedy_{self.eps}'

    @property
    def formal_name(self):
        return f'$\epsilon$-greedy, $\epsilon={self.eps}$'


class TS(AbstractPolicy):

    def __init__(self, n, gamma=1, seed=0):
        super().__init__(n, seed)
        #self.gap = gap
        self.gamma = gamma

    def select(self, regions, budget):
    
        if self.t == 0:
            samples = np.arange(budget) % len(regions)
            self.t += 1
            return Counter(samples)

        self.t += 1
        # Setup params for beta dist
        #gp = min(self.gap, len(regions[0][self.name]['cases_seen']))
        params = [(
            max(
                discount(regions[i][self.name]['cases_seen'][::-1], self.gamma),
                #np.sum(regions[i][self.name]['cases_seen'][-gp:]),
                0.1
            ),
            max(
                discount(regions[i][self.name]['n_tested'][::-1], self.gamma) -
                discount(regions[i][self.name]['cases_seen'][::-1], self.gamma),
                # np.sum(regions[i][self.name]['n_tested'][-gp:]) -
                # np.sum(regions[i][self.name]['cases_seen'][-gp:]),
                0.1
            )
        ) for i in range(len(regions))
        ]

        np.random.seed(self.seed)
        samples = np.array([beta.rvs(a, b, size=budget) for a, b in params])
        return Counter(np.argmax(samples, axis=0))

    @property
    def name(self):
        return 'TS'

    @property
    def formal_name(self):
        return self.name


class Random(AbstractPolicy):

    def __init__(self, n, seed=0):
        super().__init__(n, seed)

    def select(self, regions, budget):

        np.random.seed(self.seed)
        return Counter(
            np.random.choice(
                self.n, size=budget, replace=True
            )
        )

    def estimate_pop(self, regions, budget):
        ratio = lambda x, y: 0 if y == 0 else x/y
        return np.sum([
            ratio(v[self.name]['cases_seen'][-1], v[self.name]['n_tested'][-1])
            * v['N'] for v in regions.values()
        ])

    @property
    def name(self):
        return 'Random'

    @property
    def formal_name(self):
        return self.name
 

class Exp3(AbstractPolicy):
    
    def __init__(self, n, gamma=1, eps=0.1, seed=0):
        
        super().__init__(n, seed)
        self.eps = eps
        self.gamma = gamma
        self.w = [1 / self.n for _ in range(self.n)]
        self.prev_sample = []
        
    def select(self, regions, budget):
        
        # First we update from what we saw previously
        if self.t > 0:
            self._update(regions)

        self.t += 1
        
        # Define probability distribution 
        probs = self._pdf()
        
        # sample and record choices
        np.random.seed(self.seed)
        sample = Counter(
            np.random.choice(self.n, size=budget, p=probs)
        )
        self.prev_sample.append(sample)
        
        return Counter(sample) 
    
    def _pdf(self):

        #print(self.w)
        #self.w = np.nan_to_num(self.w, posinf=np.max([w for w in self.w if not np.isinf(w)]))
        #self.w = [np.max(w, 10**6) for w in self.w]
        #print(self.w)
        sum_w = np.sum(self.w)

        probs = [
            (1-self.eps) * (self.w[i] / sum_w) + self.eps / self.n
            for i in range(self.n)
        ]
        
        return probs 
    
    def _update(self, regions):
        
        probs = self._pdf()

        for i in range(self.n): 
            seen = regions[i][self.name]['cases_seen']
            reward = discount(seen[::-1], self.gamma)
            x = reward / probs[i]
            wi = self.w[i] * np.exp(self.eps*x / self.n)
            self.w[i] = wi

    def estimate_pop(self, regions, budget):

        pdf = self._pdf()
        return pop_est(regions, pdf, self.name, m=budget)


    @property
    def name(self):
        return f'exp3_{self.eps}'

    @property
    def formal_name(self):
        return f'Exp3, $\epsilon={self.eps}$'
        
        
        
        
        
        
        
        
        
        
        
        
    
       
    
    