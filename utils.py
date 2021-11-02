import numpy as np
from collections import Counter
from model import Ct


def discount(arr, gamma):

    return np.sum([(gamma**i) * a for i, a in enumerate(arr)])

def pop_est(regions, pdf, key, m):

    coef = lambda x, y: 0 if y == 0 else x / y
    est = np.sum([
        coef(regions[i]['N'], pdf[i]) * regions[i][key]['cases_seen'][-1]
        for i in range(len(regions))
    ])
    return est / (m * np.sum([v['N'] for v in regions.values()]))


def minmax_containment(vals, containment=0.68):
    """Returns bounding min and max values of vals, subject to given
    containment percentage.

    Args:
         vals (ndarray): nxm array, where n is number of trials, each
            with m values. Containment is considered with respect
            to number of triais.

    Returns:
        ymin, ymax: lists of length m, bounding containment % of the
            values closest to the median in vals.

    """
    median = np.median(vals, axis=0)

    bandwidth = [
        np.quantile(abs(vals[:, j] - np.median(vals[:, j])), containment)
        for j in range(vals.shape[1])
    ]

    ymin = [median[j] - bandwidth[j] for j in range(len(median))]
    ymax = [median[j] + bandwidth[j] for j in range(len(median))]

    return ymin, ymax


def sample_from_scores(scores, budget, seed=0):
    # Form pdf from scores and sample
    pdf = scores / np.sum(scores)
    np.random.seed(seed)
    return Counter(
        np.random.choice(len(scores), size=budget, p=pdf, replace=True)
    )


def gen_regions(n_regions, policies, N=1000):
    regions = {
        i: {'c0': np.random.randint(20),
            'N': N,
            'alpha': np.random.rand(1)[0]
            }
        for i in range(n_regions)
    }

    for k, v in regions.items():
        for policy in policies:
            n_tested = np.random.randint(50)
            try:
                cases_seen = np.random.randint(min(v['c0'], n_tested))
            except ValueError:
                cases_seen = 0
            regions[k][policy.name] = {
                'cases_true': [v['c0']],
                'cases_seen': [cases_seen],
                'n_tested': [n_tested]
            }

    return regions


def experiment_trial(regions, budget, beta, n_steps, policies):

    pop_ests = {
        p.name: {'est': [], 'true': []} for p in policies
    }

    for t in range(n_steps):

        samples = {}
        for policy in policies:
            samples[policy.name] = policy.select(regions, budget)

        for k in regions.keys():
            for policy in policies:
                # Update growth of disease
                regions[k][policy.name]['n_tested'].append(samples[policy.name][k])
                regions[k][policy.name]['cases_true'].append(
                    Ct(t, regions[k][policy.name]['n_tested'], regions[k]['alpha'],
                       beta, regions[k]['N'], regions[k]['c0'])
                )

                # Update cases seen
                true_prev = regions[k][policy.name]['cases_true'][-1] / regions[k]['N']
                new_cases = true_prev * samples[policy.name][k]
                regions[k][policy.name]['cases_seen'].append(new_cases)

        for policy in policies:
            try:
                # Estimated
                pop_ests[policy.name]['est'].append(
                    policy.estimate_pop(regions, budget)
                )
                # True
                pop_ests[policy.name]['true'].append(
                    np.sum([
                        v[policy.name]['cases_true'][-1] for v in regions.values()
                    ]) /
                    np.sum([
                        v['N'] for v in regions.values()
                    ])
                )
            except NotImplementedError:
                pass

    return regions, pop_ests

