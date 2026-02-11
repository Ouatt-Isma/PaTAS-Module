import numpy as np
from fractions import Fraction
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
        if(len(shape) != 2):
            raise ValueError()
        
        res = np.empty(shape=shape, dtype=TrustOpinion)
        if(value != None):
            if(not isinstance(value, TrustOpinion)):
                raise ValueError()
            for i in range(shape[0]):
                for j in range(shape[1]):
                    res[i][j] = value 
            return res 

        res = np.empty(shape=shape, dtype=TrustOpinion)
        if(method=="trust" or method=="one"):
            for i in range(shape[0]):
                for j in range(shape[1]):
                    res[i][j] = TrustOpinion.ftrust()
        elif(method=="distrust"):
            for i in range(shape[0]):
                for j in range(shape[1]):
                    res[i][j] = TrustOpinion.dtrust()
        elif(method=="vacuous"): #fill with vacuous
            for i in range(shape[0]):
                for j in range(shape[1]):
                    res[i][j] = TrustOpinion.vacuous()
        elif(method=="random"):
            for i in range(shape[0]):
                for j in range(shape[1]):
                    res[i][j] = TrustOpinion.random()
        else:
            raise ValueError(f"unsuported type of filling (method={method})")
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
            a = self.a*op2.a
            t = round(t, 20)
            d = round(d, 20)
            u = 1 -(t+d)
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
            a_fused = (a_A + a_B) / 2
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
            u = u1 * u2 / (u1 + u2 - u1 * u2)
            if ((u1 != 1) or (u2 != 1)):
                a = (a1 * u2 + a2 * u1 - (a1 + a2) * u1 * u2) / (u1 + u2 - 2 * u1 * u2)
            else:
                a = (a1 + a2) / 2
        else:
            b = 0.5 * (b1 + b2)
            u = 0
            a = 0.5 * (a1 + a2)
        ## baserate:  a ,
        ##     uncertainty: u,
        ##     belief: b,
        ##     disbelief: 1-u-b,
        ##    projectedproba: b+a*u}

        d = (1 - u - b)  ## disblief
        e = b + a * u  ## projected probability
        ## WE ROUND TO 2 DIGITS TO GET THE EXACT SAME VALUES WITH THE SIMULATION
        cf = [b, d, u, a, e]
        return TrustOpinion(b, d, u, a)
    
    def deduction_a_b():
        raise NotImplemented
    
    ##Page 150
    def p_y_x_hat(ax, b_yx, u_yx,b_ynotx,u_ynotx, ay ):
        return b_yx*ax + b_ynotx*(1-ax) + ay*(u_yx*ax + u_ynotx*(1-ax))
    
        
    #     # K = None


    #     ### NOT IN THE BOOK
    #     ### NOT IN THE BOOK 

    #     # a_yx=op_y_given_x.a

    #     # a_ynotx=op_y_given_not_x.a


    #     #K
    #     #Case 1
    #     #Case 2.A.1
    #         and (bx+ax*ux <= ax)):

        
    #     #Case 2.A.2
    #         and (bx+ax*ux > ax)):
    #             ##ux = dx = 0 
            

    #     #Case 2.B.1
    #         and (bx+ax*ux <= ax)):
        
    #     #Case 2.B.2
    #         and (bx+ax*ux > ax)):
    #     #Case 3.A.1
    #         and (bx+ax*ux <= ax)):
    #         # print(bx+ax*ux)
    #         # print((d_yx))
    #         # print(d_ynotx)

    #     #Case 3.A.2
    #         and (bx+ax*ux > ax)):

    #     #Case 3.B.1
    #         and (bx+ax*ux <= ax)):
        
    #     #Case 3.B.2
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
        return TrustOpinion(by, dy, uy, ay)
    
    
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

        return TrustOpinion.cumFuse(self, other)
    def __mul__(self, other):

        if isinstance(other, TrustOpinion):
            # Apply the multiplication to each element of the matrix
            return self.mydiscount(other) 
        else:
            # Handle other types if necessary
            return other.__mul__(self)
    def __sub__(self, other):
        return self.binMult(other)
        return TrustOpinion.avFuse(self, other)
    
        
    #     # Delegate multiplication to the other object's __mul__
         
    
    #                 sliced_arrays.append(self[idx])
    #                 sliced_arrays.append(self[idx, np.newaxis])

