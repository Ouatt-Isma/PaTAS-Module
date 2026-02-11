from concrete.TrustOpinion import TrustOpinion
import numpy as np
# from numpy.typing import NDArray

class ArrayTO:
    _DEDUCTION_VEC = np.vectorize(TrustOpinion.deduction, otypes=[TrustOpinion])
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
    
    # def __sub__(self, other: 'ArrayTO'):
    #     self_arr = self.value
    #     other_arr = other.value
    #     shape = self_arr.shape
    #     if shape == other_arr.shape:
    #         # print("sub")
    #         res = np.empty(shape=shape, dtype = TrustOpinion)
    #         for index in np.ndindex(shape):
    #             res[index]= self_arr[index] - other_arr[index] 
    #         return ArrayTO(res)
    #     else:
    #         raise ValueError("Not the same shape")
    def __sub__(self, other: 'ArrayTO'):
        self_arr = self.value
        other_arr = other.value
        return self_arr - other_arr
    
    # def __add__(self, other: 'ArrayTO'):
    #     self_arr = self.value
    #     other_arr = other.value

    #     shape = self_arr.shape
    #     if shape == other_arr.shape:
    #         # print("sub")
    #         res = np.empty(shape=shape, dtype = TrustOpinion)
    #         for index in np.ndindex(shape):
    #             res[index]= self_arr[index] + other_arr[index] 
    #         return ArrayTO(res)
    #     else:
    #         raise ValueError("Not the same shape")

    def __add__(self, other: 'ArrayTO'):
        self_arr = self.value
        other_arr = other.value
        return self_arr + other_arr
        

    def __mul__(self, other: TrustOpinion):
        if(isinstance(other, TrustOpinion)):
            shape = np.shape(self.value)
            val = np.empty(shape = shape, dtype=TrustOpinion)
            for index in np.ndindex(shape):
                val[index]= self[index]*other 
            return ArrayTO(val)
        else:
            raise ValueError("")
    # def __matmul__(self, other):
    #     """
    #     Multiply two matrices A and B using the classical algorithm.
        
    #     :param A: A list of lists where each sub-list represents a row in the first matrix.
    #     :param B: A list of lists where each sub-list represents a row in the second matrix.
    #     :return: The result of multiplying A by B as a new matrix.
    #     """
        
    #     # Get dimensions of A and B
    #     A = self.value
    #     B = other.value 
    #     A_rows, A_cols = len(A), len(A[0])
    #     B_rows, B_cols = len(B), len(B[0])
    #     # Ensure A's columns match B's rows
    #     if A_cols != B_rows:
    #         raise ValueError(f"A's columns {A_cols} must equal B's rows {B_rows} for matrix multiplication")
        
    #     # Initialize the result matrix with zeros
    #     result = [[0 for _ in range(B_cols)] for _ in range(A_rows)]
    #     result = np.empty(shape=(A_rows, B_cols), dtype=TrustOpinion)
        
    #     # Perform matrix multiplication
    #     for i in range(A_rows):
    #         # print(A[i][2])
    #         for j in range(B_cols):
    #             k = 0 
    #             # print(f"i,j:{(i,j)}= {(i,k)}*{(k,j)}")
    #             result[i][j] = A[i][k] * B[k][j]
    #             for k in range(1, A_cols):  # or B_rows, since A_cols == B_rows
    #                 # print(f"i,j+=:{(i,j)}= {(i,k)}*{(k,j)}")
    #                 result[i][j] += A[i][k] * B[k][j]
    #         # print(result[i][0])
    #     return ArrayTO(result)

    def __matmul__(self, other):
        """
        Multiply two matrices A and B using classical matrix multiplication,
        but aggregate using a custom `add` function on a list of products.
        """
        A = self.value
        B = other.value
        A_rows, A_cols = len(A), len(A[0])
        B_rows, B_cols = len(B), len(B[0])

        if A_cols != B_rows:
            raise ValueError(f"A's columns {A_cols} must equal B's rows {B_rows} for matrix multiplication")

        result = np.empty((A_rows, B_cols), dtype=TrustOpinion)
        add_opinions = TrustOpinion.add

        for i in range(A_rows):
            a_row = A[i]
            for j in range(B_cols):
                product_terms = [a_row[k] * B[k][j] for k in range(A_cols)]
                result[i][j] = add_opinions(product_terms)

        return ArrayTO(result)    
    # def __matmul__(self, other):
    #     """
    #     Multiply two matrices A and B using the classical algorithm.
        
    #     :param A: A list of lists where each sub-list represents a row in the first matrix.
    #     :param B: A list of lists where each sub-list represents a row in the second matrix.
    #     :return: The result of multiplying A by B as a new matrix.
    #     """
        
    #     # Get dimensions of A and B
    #     A = self.value
    #     B = other.value 
    #     result = A@B 
    #     return ArrayTO(result)
    

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
        # print(np.shape(loss))
        # print("loss")
        # print(loss)
        r = len(np.where(loss<epsilon_low)[0])
        s = len(loss) - r 
        W = 2
        b = r/(r+s+W)
        d = s/(r+s+W)
        u = W/(r+s+W)
        a = 0.5
        return TrustOpinion(b, d, u, a)
    
    def  theta_given_y(loss: np.array, epsilon_low: float):
        # print(np.shape(loss))
        # print("loss")
        # print(loss)
        condition = np.abs(loss) < epsilon_low
        r = np.sum(condition, axis=0)
        condition = np.abs(loss) > epsilon_low
        s = np.sum(condition, axis=0)

        # r = len(np.where(loss<epsilon_low)[0])
        # s = len(loss) - r 


        # print(r,s)

        # condition = loss == epsilon_low
        # W = 2 + np.sum(condition, axis=0)
        W = 2 
        b = r/(r+s+W)
        d = s/(r+s+W)
        u = W/(r+s+W)
        a = 0.5
        l = len(b)
        val = np.empty(shape = (1, l), dtype=TrustOpinion)
        for i in range(l):
            val[0][i] = TrustOpinion(b[i], d[i], u[i], a)
        return ArrayTO(val)
    
    def theta_given_not_y(loss, epsilon_up, mode="linear"):
        # if  mode=="classification":
        #     # Take the opposite of the y: 1-y in vector and keep the same epsilon
        #     # apply previous function
            # raise NotImplementedError
        a = 0.5
        l = np.shape(loss)[1]
        if(epsilon_up == None):
            val = np.empty(shape = (1, l), dtype=TrustOpinion)
            for i in range(l):
                # val[0][i] = TrustOpinion(b[i], d[i], u[i], a)
                val[0][i] = TrustOpinion(0, 0, 1, a)
            return ArrayTO(val)
        
        condition = np.abs(loss) > epsilon_up
        r = np.sum(condition, axis=0)
        condition = np.abs(loss) < epsilon_up
        s = np.sum(condition, axis=0)

        # condition = loss == epsilon_low
        # W = 2 + np.sum(condition, axis=0)
        W = 2
        b = r/(r+s+W)
        d = s/(r+s+W)
        u = W/(r+s+W)
        l = len(b)
        val = np.empty(shape = (1, l), dtype=TrustOpinion)
        for i in range(l):
            # val[0][i] = TrustOpinion(b[i], d[i], u[i], a)
            val[0][i] = TrustOpinion(0, 0, 1, a)
        return ArrayTO(val)

    
    # def op_theta_y(opinion_theta_given_y: 'ArrayTO', opinion_theta_given_not_y: 'ArrayTO', y: 'ArrayTO'):
    #     y_val = y.value
    #     val = np.empty(shape = np.shape(y_val), dtype=TrustOpinion)
    #     for i in range(len(y_val)):
    #         val[i] = TrustOpinion.deduction(y_val[i][0],opinion_theta_given_y, opinion_theta_given_not_y) 
    #     return ArrayTO(val)
    
    def op_theta_y_old(opinion_theta_given_y: 'ArrayTO', opinion_theta_given_not_y: 'ArrayTO', y: TrustOpinion) -> 'ArrayTO':
        # print(opinion_theta_given_y)
        # print(opinion_theta_given_not_y)
        # print(y)
        print(opinion_theta_given_y.get_shape())
        print(opinion_theta_given_not_y.get_shape())
        # print(y.get_shape())
        # raise NotImplementedError
        try:
            val = ArrayTO._DEDUCTION_VEC(y, opinion_theta_given_y.value, opinion_theta_given_not_y.value)
            return ArrayTO(val)
        except Exception as e:
            # print(e)
            try:
                return TrustOpinion.deduction(y,opinion_theta_given_y, opinion_theta_given_not_y) 
            except Exception as e:
                return ArrayTO(TrustOpinion.deduction_vectorized(y.value.T,opinion_theta_given_y.value, opinion_theta_given_not_y.value))

    def op_theta_y(opinion_theta_given_y: 'ArrayTO', opinion_theta_given_not_y: 'ArrayTO', y: 'ArrayTO') -> 'ArrayTO':
        # print(opinion_theta_given_y)
        # print(opinion_theta_given_not_y)
        # print(y)
  
        # print(y.get_shape())
        # raise NotImplementedError
        if isinstance(y, TrustOpinion):
            val = ArrayTO._DEDUCTION_VEC(y, opinion_theta_given_y.value, opinion_theta_given_not_y.value)
            return ArrayTO(val)
        else:
            val = ArrayTO._DEDUCTION_VEC(y.value.T, opinion_theta_given_y.value, opinion_theta_given_not_y.value)
            return ArrayTO(val)
            # return ArrayTO(TrustOpinion.deduction_vectorized(y.value.T,opinion_theta_given_y.value, opinion_theta_given_not_y.value))

    
    def call(self, opinion, op):
        self_arr = self.value
        op_arr = opinion.value
        shape = np.shape(self_arr)
        # print(shape)
        # print(np.shape(opinion.value))
        if(shape != np.shape(op_arr)):
            raise ValueError
        res = np.empty(shape=shape, dtype=TrustOpinion)
        for index in np.ndindex(shape):
            # res[index] = TrustOpinion.weigFuse(self.value[index],to_upd.value[index])  
            res[index] = op(self_arr[index], op_arr[index])  
            # if(self.value[index].u !=1):
            #     print("input 1")
            #     print(self.value[index])
            #     print("input 2")
            #     print(to_upd.value[index])
            #     print("output")
            #     print(res[index])
            #     raise ValueError
        return ArrayTO(res)
    

    def call_atomic(self, opinion: TrustOpinion, op: 'operator'):
        # print(type(opinion), ".........................")
        self_arr = self.value
        shape = np.shape(self_arr)
        # print(shape)
        # print(np.shape(opinion.value))
        res = np.empty(shape=shape, dtype=TrustOpinion)
        for index in np.ndindex(shape):
            # res[index] = TrustOpinion.weigFuse(self.value[index],to_upd.value[index])  
            # print(".........................",             index, ".........................")
            # print(".........................", self.value[index], ".........................")
            res[index] = op(self_arr[index], opinion)  

        return ArrayTO(res)
    
    def call_atomic_vectorized(self, opinion: 'ArrayTO', op: 'operator'):
        # print(type(opinion), ".........................")
        self_arr = self.value
        shape = np.shape(self_arr)
        # print(shape)
        # print(np.shape(opinion.value))
        res = np.empty(shape=shape, dtype=TrustOpinion)
        for index in np.ndindex(shape):
            # res[index] = TrustOpinion.weigFuse(self.value[index],to_upd.value[index])  
            print(".........................", self.value[index], ".........................")
            res[index] = op(self_arr[index], opinion)  

        return ArrayTO(res)
    


    def op_theta(weights: 'ArrayTO', opinion_theta_y: 'ArrayTO')-> 'ArrayTO' :
        """
        Revised opinon on theta based on evidence coming from y
        """
        # print(weights.get_shape())
        # print(opinion_theta_y.get_shape())
        if isinstance(opinion_theta_y, TrustOpinion):
            return weights.call_atomic(opinion_theta_y, TrustOpinion.avFuseGen)
        # if(opinion_theta_y isinstance(ArrayTO)):
        # call_atomic_vec = np.vectorize(weights.call_atomic, otypes=[TrustOpinion])
        # val = call_atomic_vec(opinion_theta_y.value, TrustOpinion.avFuse)
        # return ArrayTO(val)
        # val = np.empty(opinion_theta_y.get_shape())
        val = np.empty(weights.get_shape(), dtype=TrustOpinion)
        n,k = weights.get_shape() 
        _,m = opinion_theta_y.get_shape()
        # print(m,k,n)
        assert m == k 
        for j in range(m):
            opToF = opinion_theta_y.value[0, j]
            # print(type(opToF))
            for i in range(n):
                # print(type(val[i,j]))
                val[i,j] = TrustOpinion.avFuse(weights.value[i,j], opToF) 
                
        # for ind in np.ndindex(np.shape(opinion_theta_y.value)):
        #     tmp = weights.call_atomic(opinion_theta_y.value[ind], TrustOpinion.avFuse)

        # # 
        # # for ind in np.ndindex(np.shape(opinion_theta_y.value)):
        # #     tmp = weights.call_atomic(opinion_theta_y.value[ind], TrustOpinion.avFuse)
        # #     # print(ind[0])
        # #     val[ind[0]] = tmp.value
        return ArrayTO(val)
    

    # def update_2(self, Tx: 'ArrayTO'):
    #     # print(self.get_shape())
    #     # print(Tx.get_shape())
    #     ni1, no = self.get_shape()
    #     ni, _ = Tx.get_shape()
    #     assert ni == ni1-1, f"ni={ni}, ni1={ni1}, no={no}"
    #     res = ArrayTO(np.empty_like(self.value))
    #     for i in range(ni):
    #         for j in range(no):
    #             res[i][j] = TrustOpinion.binMult(self.value[i][j], Tx[i][0])
    #     for j in range(no):
    #         res[ni][j] = TrustOpinion.binMult(self.value[i][j], TrustOpinion.ftrust())
    #     return res 

    def update_2(self, Tx: 'ArrayTO', Ty: 'TrustOpinion'):
        # print(self.get_shape())
        # print(Tx.get_shape())
        ni1, no = self.get_shape()
        ni, _ = Tx.get_shape()
        assert ni == ni1-1, f"ni={ni}, ni1={ni1}, no={no}"
        res = ArrayTO(np.empty_like(self.value))
        for i in range(ni):
            for j in range(no):
                b = min(Tx[i][0].t, Ty.t)
                d = max(Tx[i][0].d, Ty.d)
                u = 1 -b -d 
                myOp = TrustOpinion(b,d,u)
                res[i][j] = TrustOpinion.binMult(self.value[i][j], myOp)
        for j in range(no):
            res[ni][j] = TrustOpinion.binMult(self.value[i][j], Ty)
        return res 
    
    def update_3(self, Ty: 'TrustOpinion'):
        # print(self.get_shape())
        # print(Tx.get_shape())
        ni1, no = self.get_shape()
        res = ArrayTO(np.empty_like(self.value))
        for i in range(ni1):
            for j in range(no):
                res[i][j] = TrustOpinion.avFuse(self.value[i][j], Ty)
        return res 
    # def fuse(self)->TrustOpinion:
    #     """
    #     Fuse several opinions to form one opinion. 
    #     The goal is to derive one opinion of the y batch
    #     """
    #     ## form an opinion on the batch 
      
    #     val = self.value
    #     inds = np.ndindex(np.shape(val))
    #     # print(inds)
    #     # print(type(inds))
    #     list_ind = list(inds)
    #     if (len(list_ind)==1):
    #         return val[list_ind[0]] 
    #     res = TrustOpinion.__add__(val[list_ind[0]], val[list_ind[1]])
    #     list_ind = list_ind[2:]
    #     for index in list_ind:
    #         res = TrustOpinion.__add__(res, val[index])
    #     return res 
    
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
            res = TrustOpinion.add([val[index][i] for index in range(1, bsize)])
            Arres.value[i][0] = res 
        return Arres
    
    
    def get_shape(self):
        return np.shape(self.value)
    
    def concatenate(a: 'ArrayTO', b: 'ArrayTO'):
        return ArrayTO(np.concatenate(a.value, b.value))