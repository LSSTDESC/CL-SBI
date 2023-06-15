'''
Adding subselection criteria that we may want to use to filter the simulated mass concentration pairs
that are used in SBI.

Args:
        mc_pairs: unfiltered mc_pairs from simulations
        criteria: string specifying subselection criteria

    Returns:
        mc_pairs: filtered mc_pairs
'''


def filter_mc_pairs(mc_pairs, criteria='all'):
    # TODO: what other criteria will we want to filter by?
    if criteria == 'all':
        return mc_pairs
