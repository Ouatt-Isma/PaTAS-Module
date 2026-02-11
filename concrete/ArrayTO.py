from concrete.TrustOpinion import TrustOpinion
import numpy as np

class ArrayTO:
    """Array wrapper for TrustOpinion matrices with element-wise helper operations."""
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

    def __sub__(self, other: 'ArrayTO'):
        self_arr = self.value
        other_arr = other.value
        return self_arr - other_arr

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
        condition = np.abs(loss) < epsilon_low
        r = np.sum(condition, axis=0)
        condition = np.abs(loss) > epsilon_low
        s = np.sum(condition, axis=0)




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
        a = 0.5
        l = np.shape(loss)[1]
        if(epsilon_up == None):
            val = np.empty(shape = (1, l), dtype=TrustOpinion)
            for i in range(l):
                val[0][i] = TrustOpinion(0, 0, 1, a)
            return ArrayTO(val)

        condition = np.abs(loss) > epsilon_up
        r = np.sum(condition, axis=0)
        condition = np.abs(loss) < epsilon_up
        s = np.sum(condition, axis=0)

        W = 2
        b = r/(r+s+W)
        d = s/(r+s+W)
        u = W/(r+s+W)
        l = len(b)
        val = np.empty(shape = (1, l), dtype=TrustOpinion)
        for i in range(l):
            val[0][i] = TrustOpinion(0, 0, 1, a)
        return ArrayTO(val)



    def op_theta_y_old(opinion_theta_given_y: 'ArrayTO', opinion_theta_given_not_y: 'ArrayTO', y: TrustOpinion) -> 'ArrayTO':
        try:
            val = ArrayTO._DEDUCTION_VEC(y, opinion_theta_given_y.value, opinion_theta_given_not_y.value)
            return ArrayTO(val)
        except Exception as e:
            try:
                return TrustOpinion.deduction(y,opinion_theta_given_y, opinion_theta_given_not_y)
            except Exception as e:
                return ArrayTO(TrustOpinion.deduction_vectorized(y.value.T,opinion_theta_given_y.value, opinion_theta_given_not_y.value))

    def op_theta_y(opinion_theta_given_y: 'ArrayTO', opinion_theta_given_not_y: 'ArrayTO', y: 'ArrayTO') -> 'ArrayTO':

        if isinstance(y, TrustOpinion):
            val = ArrayTO._DEDUCTION_VEC(y, opinion_theta_given_y.value, opinion_theta_given_not_y.value)
            return ArrayTO(val)
        else:
            val = ArrayTO._DEDUCTION_VEC(y.value.T, opinion_theta_given_y.value, opinion_theta_given_not_y.value)
            return ArrayTO(val)


    def call(self, opinion, op):
        """Apply a binary TrustOpinion operator element-wise with another ArrayTO."""
        self_arr = self.value
        op_arr = opinion.value
        shape = np.shape(self_arr)
        if(shape != np.shape(op_arr)):
            raise ValueError
        res = np.empty(shape=shape, dtype=TrustOpinion)
        for index in np.ndindex(shape):
            res[index] = op(self_arr[index], op_arr[index])
            # if(self.value[index].u !=1):
            #     raise ValueError
        return ArrayTO(res)


    def call_atomic(self, opinion: TrustOpinion, op: 'operator'):
        """Apply a binary TrustOpinion operator element-wise with a scalar opinion."""
        self_arr = self.value
        shape = np.shape(self_arr)
        res = np.empty(shape=shape, dtype=TrustOpinion)
        for index in np.ndindex(shape):
            res[index] = op(self_arr[index], opinion)

        return ArrayTO(res)

    def call_atomic_vectorized(self, opinion: 'ArrayTO', op: 'operator'):
        self_arr = self.value
        shape = np.shape(self_arr)
        res = np.empty(shape=shape, dtype=TrustOpinion)
        for index in np.ndindex(shape):
            print(".........................", self.value[index], ".........................")
            res[index] = op(self_arr[index], opinion)

        return ArrayTO(res)


    def op_theta(weights: 'ArrayTO', opinion_theta_y: 'ArrayTO')-> 'ArrayTO' :
        """
        Revised opinon on theta based on evidence coming from y
        """
        if isinstance(opinion_theta_y, TrustOpinion):
            return weights.call_atomic(opinion_theta_y, TrustOpinion.avFuseGen)
        val = np.empty(weights.get_shape(), dtype=TrustOpinion)
        n,k = weights.get_shape()
        _,m = opinion_theta_y.get_shape()
        assert m == k
        for j in range(m):
            opToF = opinion_theta_y.value[0, j]
            for i in range(n):
                val[i,j] = TrustOpinion.avFuse(weights.value[i,j], opToF)


        return ArrayTO(val)



    def update_2(self, Tx: 'ArrayTO', Ty: 'TrustOpinion'):
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
        ni1, no = self.get_shape()
        res = ArrayTO(np.empty_like(self.value))
        for i in range(ni1):
            for j in range(no):
                res[i][j] = TrustOpinion.avFuse(self.value[i][j], Ty)
        return res


    def fuse_batch(self) -> 'ArrayTO':
        """Fuse opinions across the batch dimension into one opinion per feature."""
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
