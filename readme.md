# Neural Networks (Delta Learning Algorithm)
## 1. Learning
Weights for a linear threshold unit can be learnt using the same methods used to set the parameters of a linear discriminant function:

* Perceptron Learning
* Minimum Squared Error Learning (Widrow-Hoff)
* Delta Learning Rule

### 1.1 Delta Learning Rule
The Delta Learning algorithm is a supervised learning approach. The weights are adjusted in proportion to the difference between:
* The desired output, __t__
* The actual output, __y__

Delta Learning can do either _sequential_ or _batch_ update:
* __Sequential__ update: __w←w+η( t−y ) x<sup>t</sup>__
* __Batch__ update: __w←w+η ∑<sub>p</sub>( t p−y p ) x p__

## 2. Delta Learning Rule pseudocode implemented using Gradient Descent:
Implementing using Gradient Descent is equivalent to the __Sequential Delta Learning Algorithm__.

* Set value of hyper-parameter __(η)__
* Initialise __w__ to arbitrary solution
* For each sample, __(x<sub>k</sub>, t<sub>k</sub>)__ in the dataset in turn:
    * Update weights: __w←w+η ( t<sub>k</sub>−H ( wx<sub>k</sub> ) ) x<sup>t</sup><sub>k</sub>__

## 3. Heaviside Function
[The Heaviside Function](https://mathworld.wolfram.com/HeavisideStepFunction.html) often written as _H(x)_, is a non continuous function whose value is zero for a negative input and one for a posotive input. In the implementation we are using the following defenition of the Heaviside Function.
<p align="center">
    <img width=auto height=auto src="https://i.imgur.com/ZFsvuGN.png">
</p>
