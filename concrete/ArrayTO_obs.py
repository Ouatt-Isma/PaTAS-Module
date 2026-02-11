from concrete.TrustOpinion import TrustOpinion
import numpy as np

class ArrayTO:
    def assert_numpy_array_with_dtype_trustopinion(obj):
        if not isinstance(obj, np.ndarray):
            raise TypeError(f"Expected object of type np.ndarray, got {type(obj)} instead.")
        if obj.dtype != np.dtype(TrustOpinion):
            raise TypeError(f"Expected dtype {TrustOpinion}, got {obj.dtype} instead.")

    def __init__(self, value: np.ndarray[TrustOpinion], ass=True):
        if(ass):
            ArrayTO.assert_numpy_array_with_dtype_trustopinion(value)
        self.value = value 
    def update(self, prev: 'ArrayTO', lr):
        return self.call(prev.call_atomic(lr, TrustOpinion.binMult), TrustOpinion.avFuse)

    def __sub__(self, other: 'ArrayTO'):
        self_arr = self.value
        other_arr = other.value
        shape = self_arr.shape
        if shape == other_arr.shape:
            res = np.empty(shape=shape, dtype = TrustOpinion)
            for index in np.ndindex(shape):
                res[index]= self_arr[index] - other_arr[index] 
            return ArrayTO(res)
        else:
            raise ValueError("Not the same shape")
    def __add__(self, other: 'ArrayTO'):
        self_arr = self.value
        other_arr = other.value

        shape = self_arr.shape
        if shape == other_arr.shape:
            res = np.empty(shape=shape, dtype = TrustOpinion)
            for index in np.ndindex(shape):
                res[index]= self_arr[index] + other_arr[index] 
            return ArrayTO(res)
        else:
            raise ValueError("Not the same shape")

    def __mul__(self, other: TrustOpinion):
        if(isinstance(other, TrustOpinion)):
            shape = np.shape(self.value)
            val = np.empty(shape = shape, dtype=TrustOpinion)
            for index in np.ndindex(shape):
                val[index]= self[index]*other 
            return ArrayTO(val)
        else:
            raise ValueError("")
    def __matmul__(self, other):
        """
        Multiply two matrices A and B using the classical algorithm.
        
        :param A: A list of lists where each sub-list represents a row in the first matrix.
        :param B: A list of lists where each sub-list represents a row in the second matrix.
        :return: The result of multiplying A by B as a new matrix.
        """
        
        # Get dimensions of A and B
        A = self.value
        B = other.value 
        A_rows, A_cols = len(A), len(A[0])
        B_rows, B_cols = len(B), len(B[0])
        # Ensure A's columns match B's rows
        if A_cols != B_rows:
            raise ValueError(f"A's columns {A_cols} must equal B's rows {B_rows} for matrix multiplication")
        
        # Initialize the result matrix with zeros
        result = [[0 for _ in range(B_cols)] for _ in range(A_rows)]
        result = np.empty(shape=(A_rows, B_cols), dtype=TrustOpinion)
        
        # Perform matrix multiplication
        for i in range(A_rows):
            for j in range(B_cols):
                k = 0 
                result[i][j] = A[i][k] * B[k][j]
                for k in range(1, A_cols):  # or B_rows, since A_cols == B_rows
                    result[i][j] += A[i][k] * B[k][j]
        return ArrayTO(result)
    def dot(A,B):
        return A.__matmul__(B)
    @property
    def T(self):
        return ArrayTO(self.value.T)
    
    def __str__(self):
        return '\n-------\n'.join(['\t'.join([str(item) for item in row]) for row in self.value])

    def __repr__(self):
        matrix_str = '-------\n '.join(['[' + ', '.join([repr(item) for item in row]) + ']' for row in self.value])
        return f'ArrayTO([\n {matrix_str}\n])'
    
    def __getitem__(self, key):
        return self.value.__getitem__(key)
    
    def theta_given_y_old(loss: np.array, epsilon_low: float):
        r = len(np.where(loss<epsilon_low)[0])
        s = len(loss) - r 
        W = 2
        b = r/(r+s+W)
        d = s/(r+s+W)
        u = W/(r+s+W)
        a = 0.5
        return TrustOpinion(b, d, u, a)
    
    def  theta_given_y(loss: np.array, epsilon_low: float):
        condition = loss < epsilon_low
        r = np.sum(condition, axis=0)
        condition = loss > epsilon_low
        s = np.sum(condition, axis=0)


        condition = loss == epsilon_low
        W = 2 + np.sum(condition, axis=0)
        b = r/(r+s+W)
        d = s/(r+s+W)
        u = W/(r+s+W)
        a = 0.5
        l = len(b)
        val = np.empty(shape = (1, l), dtype=TrustOpinion)
        for i in range(l):
            val[0][i] = TrustOpinion(b[i], d[i], u[i], a)
        return ArrayTO(val)
    
    def theta_given_not_y(loss, epsilon_up, epsilon_low=None, mode="linear"):
        #     # Take the opposite of the y: 1-y in vector and keep the same epsilon
        #     # apply previous function
        a = 0.5
        l = np.shape(loss)[1]
        if(epsilon_up == None):
            val = np.empty(shape = (1, l), dtype=TrustOpinion)
            for i in range(l):
                # val[0][i] = TrustOpinion(b[i], d[i], u[i], a)
                val[0][i] = TrustOpinion(0, 0, 1, a)
            return ArrayTO(val)
        
        condition = loss > epsilon_up
        r = np.sum(condition, axis=0)
        condition = loss < epsilon_up
        s = np.sum(condition, axis=0)

        condition = loss == epsilon_low
        W = 2 + np.sum(condition, axis=0)
        b = r/(r+s+W)
        d = s/(r+s+W)
        u = W/(r+s+W)
        l = len(b)
        val = np.empty(shape = (1, l), dtype=TrustOpinion)
        for i in range(l):
            val[0][i] = TrustOpinion(b[i], d[i], u[i], a)
            # val[0][i] = TrustOpinion(0, 0, 1, a)
        return ArrayTO(val)

    
    #         val[i] = TrustOpinion.deduction(y_val[i][0],opinion_theta_given_y, opinion_theta_given_not_y) 
    
    def op_theta_y(opinion_theta_given_y: 'ArrayTO', opinion_theta_given_not_y: 'ArrayTO', y: TrustOpinion) -> 'ArrayTO':
        
        try:
            deduction_vec = np.vectorize(TrustOpinion.deduction, otypes=[TrustOpinion])
            val = deduction_vec(y,opinion_theta_given_y.value, opinion_theta_given_not_y.value)
            return ArrayTO(val)
        except Exception as e:
            return TrustOpinion.deduction(y,opinion_theta_given_y, opinion_theta_given_not_y) 

    
    def call(self, opinion, op):
        shape = np.shape(self.value)
        if(shape != np.shape(opinion.value)):
            raise ValueError
        res = np.empty(shape=shape, dtype=TrustOpinion)
        for index in np.ndindex(np.shape(self.value)):
            # res[index] = TrustOpinion.weigFuse(self.value[index],to_upd.value[index])  
            res[index] = op(self.value[index],opinion.value[index])  
        return ArrayTO(res)
    

    def call_atomic(self, opinion: TrustOpinion, op: 'operator'):
        shape = np.shape(self.value)
        res = np.empty(shape=shape, dtype=TrustOpinion)
        for index in np.ndindex(np.shape(self.value)):
            # res[index] = TrustOpinion.weigFuse(self.value[index],to_upd.value[index])  
            res[index] = op(self.value[index],opinion)  

        return ArrayTO(res)
    
    def call_atomic_vectorized(self, opinion: 'ArrayTO', op: 'operator'):
        shape = np.shape(self.value)
        res = np.empty(shape=shape, dtype=TrustOpinion)
        for index in np.ndindex(np.shape(self.value)):
            # res[index] = TrustOpinion.weigFuse(self.value[index],to_upd.value[index])  
            print(".........................", self.value[index], ".........................")
            res[index] = op(self.value[index],opinion)  

        return ArrayTO(res)
    

    def op_theta(weights: 'ArrayTO', opinion_theta_y: 'ArrayTO')-> 'ArrayTO' :
        """
        Revised opinon on theta based on evidence coming from y
        """
        if isinstance(opinion_theta_y, TrustOpinion):
            return weights.call_atomic(opinion_theta_y, TrustOpinion.avFuse)
        val = np.empty(weights.get_shape(), dtype=TrustOpinion)
        n,k = weights.get_shape() 
        _,m = opinion_theta_y.get_shape()
        assert m == k 
        for j in range(m):
            opToF = opinion_theta_y.value[0, j]
            for i in range(n):
                val[i,j] = TrustOpinion.avFuse(weights.value[i,j], opToF) 
                

        # # 
        # # for ind in np.ndindex(np.shape(opinion_theta_y.value)):
        # #     tmp = weights.call_atomic(opinion_theta_y.value[ind], TrustOpinion.avFuse)
        # #     # print(ind[0])
        # #     val[ind[0]] = tmp.value
        return ArrayTO(val)
    

    def update_2(self, Tx: 'ArrayTO'):
        ni1, no = self.get_shape()
        ni, _ = Tx.get_shape()
        assert ni == ni1-1 
        res = ArrayTO(np.empty_like(self.value))
        for i in range(ni):
            for j in range(no):
                res[i][j] = TrustOpinion.binMult(self.value[i][j], Tx[i][0])
        for j in range(no):
            res[ni][j] = TrustOpinion.binMult(self.value[i][j], TrustOpinion.ftrust())
        return res 

    #     """
    #     Fuse several opinions to form one opinion. 
    #     The goal is to derive one opinion of the y batch
    #     """
    #     ## form an opinion on the batch 
      
    #     # print(inds)
    #     # print(type(inds))
    
    def fuse_batch(self) -> 'ArrayTO':
        """
        Fuse several opinions of a batch to form one opinion. 
        The goal is to derive one opinion of the batch
        """
        ## form an opinion on the batch 
      
        val = self.value
        bsize, n = np.shape(val)
        if bsize == 1:
            return self 
        Arres = ArrayTO(np.empty(shape = (n, 1), dtype=TrustOpinion))
        for i in range(n):
            res = TrustOpinion.__add__(val[0][i], val[1][i])
            for index in range(2, bsize):
                res = TrustOpinion.__add__(res, val[index][i])
            Arres.value[i][0] = res 
        return Arres
    
    
    def get_shape(self):
        return np.shape(self.value)
    
    def concatenate(a: 'ArrayTO', b: 'ArrayTO'):
        return ArrayTO(np.concatenate(a.value, b.value))