import torch

# Scalar
scalar1 = torch.tensor([1.])
print(scalar1)

scalar2 = torch.tensor([3.])
print(scalar2)

add_scalar = scalar1 + scalar2
print(add_scalar)

sub_scalar = scalar1 - scalar2
print(sub_scalar)

mul_scalar = scalar1 * scalar2
print(mul_scalar)

div_scalar = scalar1 / scalar2
print(div_scalar)

torch.add(scalar1, scalar2)
torch.sub(scalar1, scalar2)
torch.mul(scalar1, scalar2)
torch.div(scalar1, scalar2)

# Vector
vector1 = torch.tensor([1., 2., 3.])
print(vector1)

vector2 = torch.tensor([4., 5., 6.])
print(vector2)

add_vector = vector1 + vector2
print(add_vector)

sub_vector = vector1 - vector2
print(sub_vector)

mul_vector = vector1 * vector2
print(mul_vector)

div_vector = vector1 / vector2
print(div_vector)

torch.add(vector1, vector2)
torch.sub(vector1, vector2)
torch.mul(vector1, vector2)
torch.div(vector1, vector2)
print(torch.dot(vector1, vector2)) # 내적 1*4 + 2*5 + 3*6

# Matrix
matrix1 = torch.tensor([[1., 2.], [3., 4.]])
print(matrix1)

matrix2 = torch.tensor([[5., 6.], [7., 8.]])
print(matrix2)

sum_matrix = matrix1 + matrix2
print(sum_matrix)

sub_matrix = matrix1 - matrix2
print(sub_matrix)

mul_matrix = matrix1 * matrix2
print(mul_matrix)

div_matrix = matrix1 / matrix2
print(div_matrix)

torch.add(matrix1, matrix2)
torch.sub(matrix1, matrix2)
torch.mul(matrix1, matrix2)
torch.div(matrix1, matrix2)
print(torch.matmul(matrix1, matrix2)) # 행렬곱 연산 [[1*5 + 2*7, 1*6 + 2*8], [3*5 + 4*7, 3*6 + 4*8]]

# Tensor
tensor1 = torch.tensor([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]])
print(tensor1)

tensor2 = torch.tensor([[[9., 10.], [11., 12.]], [[13., 14.], [15., 16.]]])
print(tensor2)

sum_tensor = tensor1 + tensor2
print(sum_tensor)

sub_tensor = tensor1 - tensor2
print(sub_tensor)

mul_tensor = tensor1 * tensor2
print(mul_tensor)

div_tensor = tensor1 / tensor2
print(div_tensor)

torch.add(tensor1, tensor2)
torch.sub(tensor1, tensor2)
torch.mul(tensor1, tensor2)
torch.div(tensor1, tensor2)
print(torch.matmul(tensor1, tensor2)) # tensor size : 2 x 2 x 2