import torch
from d2l import torch as d2l



# #Activation functions decide whether a neuron should be activated or not by calculating the weighted sum and further adding bias to it. They are differentiable operators for transforming input signals to outputs, the rectified linear unit (ReLU) Given an element X the function is defined as the maximum of that element and 0, relu(x) = max(x,0).the ReLU function retains only positive elements and discards all negative elements by setting the corresponding activations to 0.
#x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True) is a tensor containing 160 elements ranging from -8 to 7.9. Since x is a tensor, we can invoke the function relu directly. The result is another tensor of the same shape containing the corresponding ReLU values. 0.1 is the step size between the elements. 
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.relu(x)
d2l.plot(x.detach(), y.detach(), 'x', 'relu(x)', figsize=(5, 2.5))
# d2l.plt.show()
#When the input is negative, the derivative of the ReLU function is 0, and when the input is positive, the derivative of the ReLU function is 1. the ReLU function is not differentiable when the input takes value precisely equal to 0.n these cases, we default to the left-hand-side derivative and say that the derivative is 0 when the input is 0
#y.backward(): This computes the gradient of y with respect to x. Here, y represents the output of the ReLU function applied to the input tensor x.
#torch.ones_like(x): The argument passed to backward() represents the gradient of some loss with respect to y. Since there is no explicit loss function here, we assume a hypothetical scalar function L where dl/dy = 1. This is why we pass torch.ones_like(x) to backward() to compute the gradient of x. for x>0, the relu function f(x)= x and its gradient(deviative) f'(x)=1. for x<0, the relu function f(x)=0 and its gradient f'(x)=0.
#The backward() function in PyTorch is crucial for training neural networks and optimizing functions in machine learning for several reasons:
# Gradient Computation
# Parameter Updates: In machine learning, especially in neural networks, we need to update the parameters (weights and biases) of our model to minimize the loss function. This process requires knowing the gradients of the loss function with respect to each parameter.
# Backpropagation
# Efficient Calculation: backward() automates the computation of these gradients through the process known as backpropagation. Backpropagation calculates the gradient of the loss function with respect to each weight by tracing the graph of computations in reverse order, from output back to inputs.
# Optimization
# Gradient Descent: Once the gradients are computed, they can be used to update the parameters using optimization algorithms such as gradient descent.
y.backward(torch.ones_like(x),retain_graph=True)
#When you call y.backward(...), PyTorch computes this derivative for each element of x and stores the result in x.grad.The result stored in x.grad after the call to y.backward() is the derivative of the sigmoid function evaluated at every point in x. 
# print(x.grad)
d2l.plot(x.detach(),x.grad,'x','grad of relu',figsize=(5,2.5))
# d2l.plt.show()

#sigmoid function The sigmoid function transforms those inputs whose values lie in the domain R, to outputs that lie on the interval (0, 1). For that reason, the sigmoid is often called a squashing function: it squashes any input in the range (-inf, inf) to some value in the range (0, 1):
# we plot the sigmoid function. Note that when the input is close to 0, the sigmoid function approaches a linear transformation.
#The sigmoid function is given by the formula f(x) = 1/(1+exp(-x)). The sigmoid function is used in the output layer of a binary classification model. It squashes the output values to the range (0,1), which can be interpreted as probabilities. The sigmoid function is differentiable, which is important for backpropagation during training.
#y = torch.sigmoid(x) computes the sigmoid function of x. The result is a tensor of the same shape as x, containing the corresponding sigmoid values.
y = torch.sigmoid(x)
print(y)
d2l.plot(x.detach(), y.detach(), 'x', 'sigmoid(x)', figsize=(5, 2.5))
# The derivative of the sigmoid function is plotted below. Note that when the input is 0, the derivative of the sigmoid function reaches a maximum of 0.25. As the input diverges from 0 in either direction, the derivative approaches 0.
x.grad.data.zero_()
y.backward(torch.ones_like(x),retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of sigmoid', figsize=(5, 2.5))
#Like the sigmoid function, the tanh (hyperbolic tangent) function also squashes its inputs, transforming them into elements on the interval between -1 and 1
y = torch.tanh(x)
d2l.plot(x.detach(), y.detach(), 'x', 'tanh(x)', figsize=(5, 2.5))