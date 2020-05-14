import math


def tanh_prime(x):
    return 1 - math.tanh(x)**2

def relu(x):
    return max(0, x)

def relu_prime(x):
    return 1 if x > 0 else 0

x = 100
a1 = math.tanh(x)
a2 = math.tanh(a1)
a3 = math.tanh(a2)
y = math.tanh(a3)

print(y)

print('w4: ', tanh_prime(a3) * a3)
print('w3: ', tanh_prime(a3) * tanh_prime(a2) * a2)
print('w2: ', tanh_prime(a3) * tanh_prime(a2) * tanh_prime(a1) * a1)
print('w1: ', tanh_prime(a3) * tanh_prime(a2) * tanh_prime(a1) * tanh_prime(x) * x)

print('----------------')

x = 100
a1 = relu(x)
a2 = relu(a1)
a3 = relu(a2)
y = relu(a3)

print(y)

print('w4: ', relu_prime(a3) * a3)
print('w3: ', relu_prime(a3) * relu_prime(a2) * a2)
print('w2: ', relu_prime(a3) * relu_prime(a2) * relu_prime(a1) * a1)
print('w1: ', relu_prime(a3) * relu_prime(a2) * relu_prime(a1) * relu_prime(x) * x)
