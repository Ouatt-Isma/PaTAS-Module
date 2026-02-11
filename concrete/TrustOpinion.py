import numpy as np
from fractions import Fraction
from typing import List
from concrete.cc import SubjectiveOpinion, cc_collection_fuse


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
            """
            binomial Multiplication of Two trust opinion self and op2
            check wheter op2 is a TrustOpinion
            """
            if( not isinstance(op2, TrustOpinion)):
                raise ValueError()

            t = self.t*op2.t + ((1-self.a)*op2.a*self.t*op2.u + (1-op2.a)*self.a*op2.t*self.u)/(1 - self.a*op2.a)
            d = self.d + op2.d - self.d*op2.d
            u = self.u*op2.u + ((1-self.a)*op2.t*self.u + (1-op2.a)*self.t*op2.u)/(1 - self.a*op2.a)
            a = op2.a
            t = round(t, 20)
            d = round(d, 20)
            u = 1 -(t+d)
            if (u<0):
                u = 0
            a = round(a, 20)
            return TrustOpinion(t, d, u, a)


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
            binomial Multiplication of Two trust opinion self and op2
            check wheter op2 is a TrustOpinion
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

    def averaging_belief_fusion(b_A, u_A, a_A, b_B, u_B, a_B):
        """
        Averages the belief fusion based on the provided belief (b), disbelief (d),
        uncertainty (u), and base rate (a) for two sources A and B.


        Parameters:
        b_A, u_A, a_A: Belief, uncertainty, and base rate for source A
        b_B, u_B, a_B: Belief, uncertainty, and base rate for source B

        Returns:
        The fused belief, uncertainty, and base rate.
        """
        # Check if both uncertainties are not zero - Case I
        if u_A != 0 or u_B != 0:
            b_fused = (b_A * u_B + b_B * u_A) / (u_A + u_B)
            u_fused = (2 * u_A * u_B) / (u_A + u_B)
            a_fused = (a_A + a_B) / 2
        # If both uncertainties are zero - Case II
        else:
            gamma_X = 1/2
            b_fused = gamma_X * b_A + (1 - gamma_X) * b_B
            u_fused = 0
            a_fused = gamma_X * a_A + (1 - gamma_X) * a_B

        return b_fused, u_fused, a_fused


    def weighted_belief_fusion(b_A, u_A, a_A, b_B, u_B, a_B):
        """
        Averages the belief fusion based on the provided belief (b), disbelief (d),
        uncertainty (u), and base rate (a) for two sources A and B.

        Parameters:
        b_A, u_A, a_A: Belief, uncertainty, and base rate for source A
        b_B, u_B, a_B: Belief, uncertainty, and base rate for source B

        Returns:
        The fused belief, uncertainty, and base rate.
        """

        # Check if both uncertainties are not zero - Case I
        if (u_A != 0 or u_B != 0) and (u_A != 1 or u_B != 1):
            b_fused = (b_A *(1-u_A)* u_B + b_B * (1-u_B)*u_A) / (u_A + u_B)
            u_fused = ((2-u_A-u_B )* u_A * u_B) / (u_A + u_B - 2*u_A*u_B)
            a_fused = (a_A*(1-u_A) + a_B*(1-u_B)) / 2
        # If both uncertainties are zero - Case II
        elif (u_A == 0 and u_B == 0):
            gamma_X = 1/2
            b_fused = gamma_X * b_A + gamma_X * b_B
            u_fused = 0
            a_fused = gamma_X * a_A + gamma_X * a_B

        elif(u_A == 1 and u_B == 1):
            gamma_X = 1/2
            b_fused = 0
            u_fused = 1
            a_fused =(a_A +  a_B)/2
        else:
            raise ValueError()
        return b_fused, u_fused, a_fused


    def avFuse(op1:'TrustOpinion', op2:'TrustOpinion'):
        assert op1.a == op2.a == 0.5
        b_A, u_A, a_A = op1.t, op1.u, op1.a
        b_B, u_B, a_B = op2.t, op2.u, op2.a
        b_fused, u_fused, a_fused = TrustOpinion.averaging_belief_fusion(b_A, u_A, a_A, b_B, u_B, a_B)
        return TrustOpinion(b_fused, 1-(b_fused+u_fused), u_fused, a_fused, precision=10, check = False)

    def weigFuse(op1:'TrustOpinion', op2:'TrustOpinion'):

        b_A, u_A, a_A = op1.t, op1.u, op1.a
        b_B, u_B, a_B = op2.t, op2.u, op2.a
        b_fused, u_fused, a_fused = TrustOpinion.weighted_belief_fusion(b_A, u_A, a_A, b_B, u_B, a_B)
        return TrustOpinion(b_fused, 1-(b_fused+u_fused), u_fused, a_fused)

    def cumFuse(op1:'TrustOpinion', op2:'TrustOpinion'):
        assert op1.a == op2.a == 0.5
        b1 = op1.t
        b2 = op2.t
        d1 = op1.d
        d2 = op2.d
        u1 = op1.u
        u2 = op2.u
        a1 = op1.a
        a2 = op2.a

        if ((u1 != 0) or (u2 != 0)):
            b = (b1 * u2 + b2 * u1) / (u1 + u2 - u1 * u2)
            u = (u1 * u2) / (u1 + u2 - u1 * u2)
            if ((u1 != 1) or (u2 != 1)):
                a = (a1 * u2 + a2 * u1 - (a1 + a2) * u1 * u2) / (u1 + u2 - 2 * u1 * u2)
            else:
                a = (a1 + a2) / 2
        else:
            b = 0.5 * (b1 + b2)
            u = 0
            a = 0.5 * (a1 + a2)

        d = (1 - u - b)  ## disblief
        if(d<0):
            d = 0
            b= 1-u
        e = b + a * u  ## projected probability
        cf = [b, d, u, a, e]
        return TrustOpinion(b, d, u, a)

    def deduction_a_b():
        raise NotImplemented

    def p_y_x_hat(ax, b_yx, u_yx,b_ynotx,u_ynotx, ay ):
        return b_yx*ax + b_ynotx*(1-ax) + ay*(u_yx*ax + u_ynotx*(1-ax))








    #         and (bx+ax*ux <= ax)):


    #         and (bx+ax*ux > ax)):


    #         and (bx+ax*ux <= ax)):

    #         and (bx+ax*ux > ax)):
    #         and (bx+ax*ux <= ax)):

    #         and (bx+ax*ux > ax)):

    #         and (bx+ax*ux <= ax)):

    #         and (bx+ax*ux > ax)):


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

        #     "baserate": ay,
        #     "uncertainty": uy,
        #     "belief": by,
        #     "disbelief": dy,
        #     "projectedproba": ey
        # }
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
        """
        Applies CC-fusion to a list of binomial opinions.
        Each opinion has keys: 'belief', 'disbelief', 'uncertainty', 'base_rate'.
        Returns a single fused binomial opinion.
        """
        beliefs = []
        disbeliefs = []
        uncertainties = []
        for op in opinions:
            beliefs.append(op.t)
            disbeliefs.append(op.d)
            uncertainties.append(op.u)
        # Step 1: Consensus
        b_cons = np.min(beliefs)
        d_cons = np.min(disbeliefs)
        bd_cons = b_cons+d_cons
        b_res = []
        d_res = []
        for op in opinions:
            b_res.append(op.t - b_cons)
            d_res.append(op.d - d_cons)


        # Step 2: Compromise
        # Compromise belief and disbelief masses

        all_u = np.prod(uncertainties)
        zero_indexes = [i for i, v in enumerate(uncertainties) if v == 0]
        zero_count = len(zero_indexes)
        if(zero_count>=2):
            b_comp = 0
            d_comp = 0
        elif(zero_count==1):
            i = zero_indexes[0]
            all_unc = 1
            for j in  range(len(opinions)):
                if(j!=i):
                    all_unc*= uncertainties[j]
            b_comp = b_res[i]*all_unc
            d_comp = d_res[i] *all_unc
        else:
            b_comp = 0
            d_comp = 0
            for i in range(len(opinions)):
                op = opinions[i]
                b_comp += b_res[i]*(all_u/op.u)
                d_comp += d_res[i]*(all_u/op.u)

            b_comp += ((0.5)**len(opinions))*np.prod(b_res) ## Assuming a always set to 0.5\
            d_comp += ((0.5)**len(opinions))*np.prod(d_res) ## Assuming a always set to 0.5

        u_pre = all_u
        bd_comp = b_comp + d_comp
        print(b_cons,b_comp,u_pre)
        print(d_cons,d_comp,u_pre)
        print(bd_cons+bd_comp+u_pre)
        eta = (1-bd_cons - u_pre)/(2*bd_comp)
        print(eta)
        u = u_pre + eta*np.min(uncertainties)
        b = b_cons + eta*b_comp
        d = d_cons + eta*d_comp
        norm = b+d+u

        return TrustOpinion(b/norm,d/norm,u/norm)
    def cc_fusion_binomial(opinions: List['TrustOpinion']) -> 'TrustOpinion':

        all_equal = all(v == 0.5 for v in [op.a for op in opinions])
        if (not all_equal):
            raise ValueError("Not all base rate are equals to 0.5")
        opinions = [
            SubjectiveOpinion(op.t, op.d, op.u, op.a) for op in opinions
        ]
        fused = cc_collection_fuse(opinions)
        return TrustOpinion(fused.belief, fused.disbelief, fused.uncertainty, fused.base_rate)

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
