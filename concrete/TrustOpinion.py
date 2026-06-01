import numpy as np
from fractions import Fraction
from typing import List

from patas_module.subjective_logic import (
    Opinion,
    averaging_fusion,
    cumulative_fusion,
    weighted_belief_fusion,
    consensus_compromise_fusion,
    multiply,
)


# ── Bridge helpers ────────────────────────────────────────────────────────────

def _to_op(to: 'TrustOpinion') -> Opinion:
    """TrustOpinion → Opinion  (trust=b, same d/u/a)."""
    return Opinion(to.t, to.d, to.u, to.a)


def _from_op(op: Opinion) -> 'TrustOpinion':
    """Opinion → TrustOpinion  (b→trust, same d/u/a)."""
    return TrustOpinion(op.b, op.d, op.u, op.a, precision=20, check=False)


class TrustOpinion:
    """
    A subjective trust opinion define as belief(trust), disbelief(trust), uncertainty(untrust) and base rate
    """
    def __init__(self, trust_mass, distrust_mass, untrust_mass, base_rate=0.5, precision = 4, check=True):
        """
        Instantiate a trust opinion object.
        First check that all input are positive
        Second check whether the sum of trust_mass, distrust_mass and untrust_mass are equal to 1
        Third check that the base rate are less than
        """

        trust_mass = round(trust_mass, precision)
        distrust_mass = round(distrust_mass, precision)
        untrust_mass = round(untrust_mass, precision)
        if(check):
            assert trust_mass >= 0 and distrust_mass >= 0 and untrust_mass >= 0 and base_rate >= 0, \
                f"Invalid values: trust_mass={trust_mass}, distrust_mass={distrust_mass}, untrust_mass={untrust_mass}, base_rate={base_rate}"

            assert round(trust_mass + distrust_mass + untrust_mass, precision - 1) == 1, \
                f"Sum of masses does not equal 1: trust_mass={trust_mass}, distrust_mass={distrust_mass}, untrust_mass={untrust_mass}"
            assert base_rate <= 1, f"Base rate exceeds 1: base_rate={base_rate}"


        self.t = trust_mass
        self.d = distrust_mass
        self.u = untrust_mass
        self.a = base_rate


    def equalTo(self, what):
        if (what == "trust"):
            if self.t == 1:
                return True
            return False
        if (what == "vacuous"):
            if self.u == 1:
                return True
            return False
        if (what == "distrust"):
            if self.d == 1:
                return True
            return False
        raise NotImplementedError

    def vacuous():
        """
        returns a vacuous trust opinion (0, 0, 1)
        """
        return TrustOpinion(0, 0, 1)

    def ftrust():
        """
        returns a fully trust opinion (1, 0, 0)
        """
        return TrustOpinion(1, 0, 0)
    def dtrust():
        """
        returns a fully distrust opinion (0, 1, 0)
        """
        return TrustOpinion(0, 1, 0)
    def generate_biased_triplet(bias_index=0, bias_strength=0.6):
        """
        Generate three positive numbers that sum to 1,
        with one of them arbitrarily larger.

        Parameters:
        - bias_index: which index (0, 1, or 2) is to be larger
        - bias_strength: how large it should be (e.g. 0.7 means it will take ~70% of total)

        Returns:
        - A list of 3 numbers summing to 1
        """
        assert 0 < bias_strength < 1, "Bias strength must be between 0 and 1"
        assert bias_index in [0, 1, 2], "Bias index must be 0, 1, or 2"

        # The remaining sum to be split among the other two
        remaining = 1 - bias_strength
        other_two = np.random.dirichlet([1, 1]) * remaining

        result = [0, 0, 0]
        result[bias_index] = bias_strength

        idx = [i for i in range(3) if i != bias_index]
        result[idx[0]] = other_two[0]
        result[idx[1]] = other_two[1]

        return result

    def generate_biased_opinion(method):
        if (method=="trust"):
            result = TrustOpinion.generate_biased_triplet()
        elif (method=="distrust"):
            result = TrustOpinion.generate_biased_triplet(bias_index=1)
        elif (method=="vacuous"):
            result = TrustOpinion.generate_biased_triplet(bias_index=2)
        else:
            raise NotImplementedError
        return TrustOpinion(result[0], result[1], result[2])

    def generate_biased_matrix(method, size):
        m,n = size
        return np.array([ [TrustOpinion.generate_biased_opinion(method) for i in range(m)] for j in range(n)], dtype=TrustOpinion)

    def generate_biased_vector(method, size):

        return np.array([TrustOpinion.generate_biased_opinion(method) for i in range(size)], dtype=TrustOpinion)

    def random(n=10):
        """
        Returns a random trust opinion
        bigger n is, the bigger the set of possible random opinion is
        """

        a,b,c = np.random.randint(1, n, 3)
        t = a/(a+b+c)
        d = b/(a+b+c)
        u = c/(a+b+c)

        return TrustOpinion(t, d, u)

    def random_matrix(n, m):
        """
        Returns an array of random trust opinion
        """
        return np.array([ [TrustOpinion.random() for i in range(m)] for j in range(n)], dtype=TrustOpinion)

    def fill(shape, method="random", value: 'TrustOpinion' =None):
        """
        Create an ArrayTO (Array of opinion) that has the shape _shape.
        The filling depends on the method:
        -random: randomly
        -trust or one: fully trust opinion (1,0,0)
        -vacuous: Vacuous opinio (0,0,1)
        -distrust: (0,1,0)
        -if value is set, then the all entry will be set to value.

        First Check that the shape is a tuple of size 2
        Second if value is not set to None, check that it's an instance of trust opinion
        """
        if len(shape) != 2:
            raise ValueError()

        res = np.empty(shape=shape, dtype=TrustOpinion)
        if value is not None:
            if not isinstance(value, TrustOpinion):
                raise ValueError()
            for index in np.ndindex(shape):
                res[index] = value
            return res

        factories = {
            "trust": TrustOpinion.ftrust,
            "distrust": TrustOpinion.dtrust,
            "vacuous": TrustOpinion.vacuous,
            "one": TrustOpinion.vacuous,
            "random": TrustOpinion.random,
            "vacuous2": lambda: TrustOpinion(0.25, 0.25, 0.5),
        }

        try:
            factory = factories[method]
        except KeyError as exc:
            raise ValueError(f"unsuported type of filling (method={method})") from exc

        for index in np.ndindex(shape):
            res[index] = factory()
        return res


    def projected_prob(self, frac = False):
        """
        Returns projected probability of the trust opinion
        """
        if frac:
            return Fraction(self.t + self.a*self.u ).limit_denominator()
        return round(self.t + self.a*self.u,3)

    def print(self, frac=False):
        """
        print the opinion
        set frac to True if one want to print each fields as a fraction
        """
        res = "({},{},{})".format(round(self.t,3), round(self.d,3), round(self.u,3))
        if (frac):
            res = "({},{},{},{})".format(Fraction(self.t).limit_denominator(), Fraction(self.d).limit_denominator(), Fraction(self.u).limit_denominator(), Fraction(self.a).limit_denominator())

        return res

    def binMult(self, op2):
        """Binomial multiplication ⊙ — delegates to subjective_logic.multiply."""
        if not isinstance(op2, TrustOpinion):
            raise ValueError()
        return _from_op(multiply(_to_op(self), _to_op(op2)))


    def mydiscount(self, op2):
            """
            binomial Multiplication of Two trust opinion self and op2
            check wheter op2 is a TrustOpinion
            """
            if( not isinstance(op2, TrustOpinion)):
                raise ValueError()

            t = self.t*op2.t
            d = self.d*op2.t + op2.d*self.t
            u = 1 -(t+d)
            a = self.a
            t = round(t, 20)
            d = round(d, 20)
            u = 1 -(t+d)
            a = round(a, 20)
            return TrustOpinion(t, d, u, a)

    def discount(self, op2):
            """
            Probability-sensitive trust discounting.
            Uses projected probability p = t + u·a as the discount factor.
            (Note: this is the projected-probability variant; the canonical
            subjective_logic.discount() uses belief mass only.)
            """
            if( not isinstance(op2, TrustOpinion)):
                raise ValueError()

            p = self.t+self.u*self.a
            t = p*op2.t
            d = p*op2.d
            u = 1 -(t+d)
            a = self.a
            t = round(t, 20)
            d = round(d, 20)
            u = 1 -(t+d)
            a = round(a, 20)
            return TrustOpinion(t, d, u, a)

    def binomialMultiplication(op1: 'TrustOpinion', op2: 'TrustOpinion'):
        """
        The only difference with binMult is that here the function is static.
        Check whether op1 is a TrustOpinion
        """
        if( not isinstance(op1, TrustOpinion)):
                raise ValueError()
        return op1.binMult(op2)

    def avFuse(op1:'TrustOpinion', op2:'TrustOpinion'):
        """Averaging belief fusion (ABF) ⊕_avg — delegates to subjective_logic.averaging_fusion."""
        assert op1.a == op2.a == 0.5
        return _from_op(averaging_fusion(_to_op(op1), _to_op(op2)))

    def weigFuse(op1:'TrustOpinion', op2:'TrustOpinion'):
        """Weighted belief fusion (WBF) ⊕_w — delegates to subjective_logic.weighted_belief_fusion."""
        return _from_op(weighted_belief_fusion([_to_op(op1), _to_op(op2)]))

    def cumFuse(op1:'TrustOpinion', op2:'TrustOpinion'):
        """Cumulative belief fusion (CBF) ⊕_c — delegates to subjective_logic.cumulative_fusion."""
        assert op1.a == op2.a == 0.5
        return _from_op(cumulative_fusion(_to_op(op1), _to_op(op2)))

    def deduction_a_b():
        raise NotImplemented

    def p_y_x_hat(ax, b_yx, u_yx,b_ynotx,u_ynotx, ay ):
        return b_yx*ax + b_ynotx*(1-ax) + ay*(u_yx*ax + u_ynotx*(1-ax))


    def adjust(a):
        if(a == 0):
            a+=10^-5
        if(a == -1 ):
            a-=10^-5
        return a
    def deduction(op_x: 'TrustOpinion', op_y_given_x: 'TrustOpinion', op_y_given_not_x: 'TrustOpinion',debug=False):

        if(round(op_x.t,3) == 1):
            return op_y_given_x
        elif(round(op_x.d,3) == 1):
            return op_y_given_not_x

        if debug:
            print("debug")
            print(op_x.d)
            print(op_x.d == 1)
            print(op_y_given_x)
            print(op_y_given_not_x)
        ax = op_x.a
        bx = op_x.t
        dx = op_x.d
        ux = op_x.u
        ex = op_x.projected_prob()


        b0 = op_y_given_x.t
        d0 = op_y_given_x.d
        u0 = op_y_given_x.u
        e0 = op_y_given_x.projected_prob()

        b1 = op_y_given_not_x.t
        d1 = op_y_given_not_x.d
        u1 = op_y_given_not_x.u
        e1 = op_y_given_not_x.projected_prob()

        ay = (ax*b0 + (1-ax)*b1)/(1 - ax*u0 - (1-ax)*u0)

        bIy = bx * b0 + dx * b1 + ux * (b0 * ax + b1 * (1 - ax))
        dIy = bx * d0 + dx * d1 + ux * (d0 * ax + d1 * (1 - ax))
        uIy = bx * u0 + dx * u1 + ux * (u0 * ax + u1 * (1 - ax))
        Pyvacuousx = b0 * ax + b1 * (1 - ax) + ay * (u0 * ax + u1 * (1 - ax))

        K = 0


        if ((b0 > b1) and (d0 > d1)) or ((b0 <= b1) and (d0 <= d1)):  # CASE I
            K = 0
        elif (b0 > b1) and (d0 <= d1):  # CASE II
            if Pyvacuousx <= (b1 + ay * (1 - b1 - d0)):  # CASE A
                if ex <= ax:  # Case 1
                    K = ax * ux * (bIy - b1) / (ay * ex)
                else:  # Case 2
                    K = ax * ux * (dIy - d0) * (b0 - b1) / ((dx + (1 - ax) * ux) * ay * (d1 - d0))
            else:  # CASE B
                if ex <= ax:  # Case 1
                    K = (1 - ax) * ux * (bIy - b1) * (d1 - d0) / (ex * (1 - ay) * (b0 - b1))
                else:  # Case 2
                    K = (1 - ax) * ux * (dIy - d0) / ((1 - ay) * (dx + (1 - ax) * ux))
        else:  # CASE III
            if Pyvacuousx <= (b1 + ay * (1 - b1 - d0)):  # CASE A
                if ex <= ax:  # Case 1
                    K = (1 - ax) * ux * (dIy - d1) * (b1 - b0) / (ex * ay * (d0 - d1))
                else:  # Case 2
                    K = (1 - ax) * ux * (bIy - b0) / (ay * (dx + (1 - ax) * ux))
            else:  # CASE B
                if ex <= ax:  # Case 1
                    K = ax * ux * (dIy - d1) / (ex * (1 - ay))
                else:  # Case 2
                    K = ax * ux * (bIy - b0) * (d0 - d1) / ((1 - ay) * (b1 - b0) * (dx + (1 - ax) * ux))

        if K is None or not isinstance(K, float) or not (K == K):  # Check for NaN
            K = 0

        by = bIy - ay * K
        dy = dIy - (1 - ay) * K
        uy = uIy + K
        ey = by + ay * uy

        return TrustOpinion(by, dy, uy, 0.5)


    def deduction_vectorized(op_x_arr: np.ndarray, op_y_given_x_arr: np.ndarray, op_y_given_not_x_arr: np.ndarray):
        """
        Vectorized deduction operation on arrays of TrustOpinion objects.
        All input arrays must be of the same shape and type np.ndarray.
        Returns a np.ndarray of TrustOpinion objects.
        """
        print(op_x_arr.shape)
        print(op_y_given_x_arr.shape )
        print(op_y_given_not_x_arr.shape)
        assert op_x_arr.shape == op_y_given_x_arr.shape == op_y_given_not_x_arr.shape, f"Arrays must be same shape {op_x_arr.shape},{op_y_given_x_arr.shape},{op_y_given_not_x_arr.shape}"

        result = np.empty_like(op_x_arr)

        for i in range(op_x_arr.shape[0]):
            op_x = op_x_arr[i][0]
            op_y_given_x = op_y_given_x_arr[i][0]
            op_y_given_not_x = op_y_given_not_x_arr[i][0]

            if round(op_x.t, 3) == 1:
                result[i] = op_y_given_x
                continue
            elif round(op_x.d, 3) == 1:
                result[i] = op_y_given_not_x
                continue

            ax = op_x.a
            bx = op_x.t
            dx = op_x.d
            ux = op_x.u
            ex = op_x.projected_prob()

            b0 = op_y_given_x.t
            d0 = op_y_given_x.d
            u0 = op_y_given_x.u
            e0 = op_y_given_x.projected_prob()

            b1 = op_y_given_not_x.t
            d1 = op_y_given_not_x.d
            u1 = op_y_given_not_x.u
            e1 = op_y_given_not_x.projected_prob()

            denominator = 1 - ax * u0 - (1 - ax) * u0
            ay = (ax * b0 + (1 - ax) * b1) / denominator if denominator != 0 else 0.5

            bIy = bx * b0 + dx * b1 + ux * (b0 * ax + b1 * (1 - ax))
            dIy = bx * d0 + dx * d1 + ux * (d0 * ax + d1 * (1 - ax))
            uIy = bx * u0 + dx * u1 + ux * (u0 * ax + u1 * (1 - ax))
            Pyvacuousx = b0 * ax + b1 * (1 - ax) + ay * (u0 * ax + u1 * (1 - ax))

            K = 0
            try:
                if ((b0 > b1 and d0 > d1) or (b0 <= b1 and d0 <= d1)):
                    K = 0
                elif (b0 > b1 and d0 <= d1):
                    if Pyvacuousx <= (b1 + ay * (1 - b1 - d0)):
                        K = ax * ux * (bIy - b1) / (ay * ex) if ex <= ax else ax * ux * (dIy - d0) * (b0 - b1) / ((dx + (1 - ax) * ux) * ay * (d1 - d0))
                    else:
                        K = (1 - ax) * ux * (bIy - b1) * (d1 - d0) / (ex * (1 - ay) * (b0 - b1)) if ex <= ax else (1 - ax) * ux * (dIy - d0) / ((1 - ay) * (dx + (1 - ax) * ux))
                else:
                    if Pyvacuousx <= (b1 + ay * (1 - b1 - d0)):
                        K = (1 - ax) * ux * (dIy - d1) * (b1 - b0) / (ex * ay * (d0 - d1)) if ex <= ax else (1 - ax) * ux * (bIy - b0) / (ay * (dx + (1 - ax) * ux))
                    else:
                        K = ax * ux * (dIy - d1) / (ex * (1 - ay)) if ex <= ax else ax * ux * (bIy - b0) * (d0 - d1) / ((1 - ay) * (b1 - b0) * (dx + (1 - ax) * ux))
            except:
                K = 0

            if K is None or not isinstance(K, float) or not (K == K):
                K = 0

            by = bIy - ay * K
            dy = dIy - (1 - ay) * K
            uy = uIy + K

            result[i] = TrustOpinion(by, dy, uy, ay)

        return result


    def test_deduction():

        op_yx = TrustOpinion(0.57, 0.1, 0.33, 0.34)
        op_ynotx = TrustOpinion(0, 1, 0, 0.34)
        op_x = TrustOpinion(0.46, 0.2, 0.34, 0.5)

        print(op_yx.print())
        print(op_ynotx.print())
        print(op_x.print())

        d= TrustOpinion.deduction(op_x,op_yx,op_ynotx)

        print(d.print())

    def __str__(self):
        return self.print()
    def __rep__(self):
        return self.print()

    def __add__(self, other):
        raise NotImplementedError

    def add(opinions: List['TrustOpinion']) -> 'TrustOpinion':
        return TrustOpinion.avFuseGen(opinions)


    def __mul__(self, other):

        if isinstance(other, TrustOpinion):
            # Apply the multiplication to each element of the matrix
            return other.discount(self)
        else:
            # Handle other types if necessary
            return other.__mul__(self)
    def __sub__(self, other):
        return self.binMult(other)
        return TrustOpinion.avFuse(self, other)



    def MyCCFuse(opinions: List['TrustOpinion']) -> 'TrustOpinion':
        """Consensus & Compromise Fusion (CCF) — delegates to subjective_logic.consensus_compromise_fusion."""
        return _from_op(consensus_compromise_fusion(_to_op(op) for op in opinions))

    def cc_fusion_binomial(opinions: List['TrustOpinion']) -> 'TrustOpinion':
        """CCF for binomial opinions with base rate 0.5 — delegates to subjective_logic.consensus_compromise_fusion."""
        all_equal = all(v == 0.5 for v in [op.a for op in opinions])
        if not all_equal:
            raise ValueError("Not all base rate are equals to 0.5")
        return _from_op(consensus_compromise_fusion(_to_op(op) for op in opinions))

    def MyCons2Fuse(opinions: List['TrustOpinion']) -> 'TrustOpinion':
        all_equal = all(v == 0.5 for v in [op.a for op in opinions])
        if (not all_equal):
            raise ValueError("Not all base rate are equals to 0.5")

        b = np.min([op.t for op in opinions])
        d = np.min([op.d for op in opinions])
        u = np.min([op.u for op in opinions])

        if(b==d and d==u):
            return TrustOpinion(1/3,1/3,1/3)
        tot = b+d+u
        return TrustOpinion(b/tot, d/tot, u/tot, 0.5)

    def normalize_opinion(t, d, u, precision=4):
        total = t + d + u
        return round(t / total, precision), round(d / total, precision), round(u / total, precision)

    def avFuseGen(opinions: List['TrustOpinion']) -> 'TrustOpinion':
        """Arithmetic mean of opinion masses (not ABF). Used for batch aggregation."""
        n = len(opinions)
        t = sum(op.t for op in opinions) / n
        d = sum(op.d for op in opinions) / n
        u = sum(op.u for op in opinions) / n
        t, d, u = TrustOpinion.normalize_opinion(t, d, u)
        return TrustOpinion(t, d, u, base_rate=0.5)

    def cumFuseGen(opinions: List['TrustOpinion']) -> 'TrustOpinion':
        t = 1
        d = 1
        u = 1
        for op in opinions:
            t *= op.t
            d *= op.d
            u *= op.u
        t, d, u = TrustOpinion.normalize_opinion(t, d, u)
        return TrustOpinion(t, d, u, base_rate=0.5)

    def weigFuseGen(opinions: List['TrustOpinion']) -> 'TrustOpinion':
        weights = [1-opinions[i] for i in range(len(opinions))]
        assert len(opinions) == len(weights), "Mismatched opinions and weights"
        total_weight = sum(weights)
        t = sum(op.t * w for op, w in zip(opinions, weights)) / total_weight
        d = sum(op.d * w for op, w in zip(opinions, weights)) / total_weight
        u = sum(op.u * w for op, w in zip(opinions, weights)) / total_weight
        t, d, u = TrustOpinion.normalize_opinion(t, d, u)
        return TrustOpinion(t, d, u, base_rate=0.5)

    def belief_constraint_fusion(opinions, epsilon=0.1):
        avg_belief = sum(op.t for op in opinions) / len(opinions)
        for op in opinions:
            if abs(op.t - avg_belief) > epsilon:
                raise ValueError("Belief disagreement exceeds threshold ε")
        return TrustOpinion.avFuseGen(opinions)
