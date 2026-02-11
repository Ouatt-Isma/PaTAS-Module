








import numpy as np

class CustomDataType:
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    def __matmul__(self, other):
        if not isinstance(other, CustomDataType):
            raise ValueError("Operand must be an instance of CustomDataType")

        # Define how the dot product is computed. This is just an example.
        return self.a * other.a + self.b * other.b + self.c * other.c

    def __repr__(self):
        return f"CustomDataType(a={self.a}, b={self.b}, c={self.c})"

# Example usage:
element1 = CustomDataType(1, 2, 3)
element2 = CustomDataType(4, 5, 6)

# Using the @ operator for the dot product
result = element1 @ element2
print("Dot product:", result)

# For arrays of these custom objects, you'll need to manually handle operations
arr1 = np.array([CustomDataType(1, 2, 3), CustomDataType(4, 5, 6)])
arr2 = np.array([CustomDataType(7, 8, 9), CustomDataType(10, 11, 12)])

# Manually computing dot product for arrays of custom objects
dot_products = np.array([a @ b for a, b in zip(arr1, arr2)])
print("Dot products for arrays:", dot_products)
