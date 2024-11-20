# import torch

# x = torch.tensor([7.0], requires_grad=True)
# print("x: ", x)
# print("x.grad: ", x.grad)  # Will print None since no backward pass has been performed yet

# y = x * 2  # x_grad was undefined, should use x directly
# print("y: ", y)

# y.backward()
# print("x.grad: ", x.grad)  # Now will print the gradient

import torch

# x를 requires_grad=True로 설정하여 gradient를 추적
x = torch.tensor([2.0, 3.0], requires_grad=True)

# 간단한 연산
y = x[0]**2 + x[1]**3  # y = x₀² + x₁³

# 역전파 수행
y.backward()

# x의 gradient 출력
print("x.grad: ", x.grad)  # tensor([4.0, 27.0])