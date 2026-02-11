from itertools import product
from functools import reduce
import numpy as np


class OpinionArithmeticException(Exception):
    pass


class SubjectiveOpinion:
    def __init__(self, belief, disbelief, uncertainty, base_rate):
        self.belief = belief
        self.disbelief = disbelief
        self.uncertainty = uncertainty
        self.base_rate = base_rate

    def check_consistency(self, strict=False):
        total = self.belief + self.disbelief + self.uncertainty
        if strict and not np.isclose(total, 1.0):
            raise ValueError("Belief, disbelief, and uncertainty must sum to 1")

    def get_belief(self):
        return self.belief

    def get_disbelief(self):
        return self.disbelief

    def get_uncertainty(self):
        return self.uncertainty

    def get_base_rate(self):
        return self.base_rate


# Domain definitions
class Domain:
    TRUE = "TRUE"
    FALSE = "FALSE"
    DOMAIN = "DOMAIN"
    NIL = "NIL"

    @staticmethod
    def intersect(domains):
        if Domain.NIL in domains:
            return Domain.NIL
        if all(d == Domain.TRUE for d in domains):
            return Domain.TRUE
        if all(d == Domain.FALSE for d in domains):
            return Domain.FALSE
        return Domain.NIL

    @staticmethod
    def union(domains):
        if Domain.DOMAIN in domains:
            return Domain.DOMAIN
        if all(d == Domain.TRUE for d in domains):
            return Domain.TRUE
        if all(d == Domain.FALSE for d in domains):
            return Domain.FALSE
        return Domain.DOMAIN if any(d in [Domain.TRUE, Domain.FALSE] for d in domains) else Domain.NIL


def cc_collection_fuse(opinions):
    if any(o is None for o in opinions) or len(opinions) < 2:
        raise ValueError("Cannot fuse null opinions or fuse less than two opinions.")

    base_rate = opinions[0].get_base_rate()
    if any(o.get_base_rate() != base_rate for o in opinions):
        raise OpinionArithmeticException("Base rates must be the same.")

    consensus_belief = min(o.get_belief() for o in opinions)
    consensus_disbelief = min(o.get_disbelief() for o in opinions)
    consensus_mass = consensus_belief + consensus_disbelief

    residue_beliefs = [max(o.get_belief() - consensus_belief, 0) for o in opinions]
    residue_disbeliefs = [max(o.get_disbelief() - consensus_disbelief, 0) for o in opinions]
    uncertainties = [o.get_uncertainty() for o in opinions]
    product_uncertainty = np.prod(uncertainties)
    zero_indexes = [i for i, v in enumerate(uncertainties) if v == 0]
    zero_count = len(zero_indexes)
    if(zero_count>=2):
        comp_belief = 0
        comp_disbelief = 0
    elif(zero_count==1):
        i = zero_indexes[0]
        all_unc = 1
        for j in  range(len(opinions)):
            if(j!=i):
                all_unc*= uncertainties[j]
        comp_belief = residue_beliefs[i]*all_unc
        comp_disbelief = residue_disbeliefs[i] *all_unc
    else:
        comp_belief = sum(residue_beliefs[i] * (product_uncertainty / uncertainties[i]) for i in range(len(opinions)))
        comp_disbelief = sum(residue_disbeliefs[i] * (product_uncertainty / uncertainties[i]) for i in range(len(opinions)))
    comp_x = 0

    domain_values = [Domain.TRUE, Domain.FALSE, Domain.DOMAIN, Domain.NIL]
    for permutation in product(domain_values, repeat=len(opinions)):
        intersection = Domain.intersect(permutation)
        union = Domain.union(permutation)

        if intersection == Domain.TRUE:
            if Domain.DOMAIN not in permutation:
                prod = np.prod([residue_beliefs[j] for j in range(len(permutation)) if permutation[j] == Domain.TRUE])
                comp_belief += prod

        elif intersection == Domain.FALSE:
            if Domain.DOMAIN not in permutation:
                prod = np.prod([residue_disbeliefs[j] for j in range(len(permutation)) if permutation[j] == Domain.FALSE])
                comp_disbelief += prod

        if union == Domain.DOMAIN and intersection == Domain.NIL:
            prod = 1
            for j in range(len(permutation)):
                if permutation[j] == Domain.TRUE:
                    prod *= residue_beliefs[j]
                elif permutation[j] == Domain.FALSE:
                    prod *= residue_disbeliefs[j]
                else:
                    prod = 0
            comp_x += prod

        elif union == Domain.TRUE and intersection == Domain.NIL:
            prod = 1
            for j in range(len(permutation)):
                if permutation[j] == Domain.TRUE:
                    prod *= residue_beliefs[j]
                elif permutation[j] == Domain.FALSE:
                    prod *= residue_disbeliefs[j]
                elif permutation[j] == Domain.NIL:
                    prod = 0
            comp_belief += prod

        elif union == Domain.FALSE and intersection == Domain.NIL:
            prod = 1
            for j in range(len(permutation)):
                if permutation[j] == Domain.TRUE:
                    prod *= residue_beliefs[j]
                elif permutation[j] == Domain.FALSE:
                    prod *= residue_disbeliefs[j]
                elif permutation[j] == Domain.NIL:
                    prod = 0
            comp_disbelief += prod

    comp_mass = comp_belief + comp_disbelief + comp_x
    norm_factor = (1 - consensus_mass - product_uncertainty) / comp_mass if comp_mass != 0 else 0

    fused_belief = consensus_belief + norm_factor * comp_belief
    fused_disbelief = consensus_disbelief + norm_factor * comp_disbelief
    fused_uncertainty = product_uncertainty + norm_factor * comp_x
    if(comp_mass==0):
        tot = (1 - (fused_belief+fused_disbelief+fused_uncertainty))/3
        fused_uncertainty+= tot
        fused_belief+=tot
        fused_disbelief+=tot
    result = SubjectiveOpinion(fused_belief, fused_disbelief, fused_uncertainty, base_rate)
    result.check_consistency(strict=True)
    return result


#     SubjectiveOpinion(0, 1, 0, 0.5),
#     SubjectiveOpinion(1, 0, 0, 0.5),
