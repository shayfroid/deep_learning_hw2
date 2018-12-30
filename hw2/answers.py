r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 2 answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    # raise NotImplemented
    wstd = 0.1
    lr = 0.03
    reg = 0.01
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = 0, 0, 0, 0, 0

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    wstd = 0.1
    lr_vanilla = 0.01
    reg = 0.0001
    lr_momentum = 0.0015
    lr_rmsprop =0.0001
    # ========================
    return dict(wstd=wstd, lr_vanilla=lr_vanilla, lr_momentum=lr_momentum,
                lr_rmsprop=lr_rmsprop, reg=reg)


def part2_dropout_hp():
    wstd, lr, = 0, 0
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    wstd = 0.1
    lr = 5e-4
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Your answer:**
The key features of the results match what we expected to see. With no dropout and such a small training set we
obviously overfit the training data very fast and get good train accuracy. As expected with overfitted data, the test
results are not impressive.

For dropout = 0.4: it fits what we expected to see. the dropout acts as an overfit remedy and the results are more balanced
with 71% acc for train and 27% for test. Here we expected to see a bt more improvement in the test acc result but we
assume the small test accuracy improvement is arising from not tuned enough hiperparameters.

for dropout of 0.8: exactly as expected. 0.8 means that with 80% chance we "forget" everything that we learn
so a big drop in train accuracy was expected (dropped to 30%).
The drop of test was also expected since we effectively have no learning what so ever.

"""

part2_q2 = r"""
**Your answer:**

There is a possibility that the classification probability for the samples which are not predicted (and are wrong) increases.
For example if a picture of a horse that is predicted 0.4 horse and 0.3 dog in the first epoch will be predicted 0.8 horse and 0.5 dog
in the second epoch, the loss will be higher and the accuracy will stay the same, if you add in a cat which was predicted 0.5 dog and 0.4 cat
in the first epoch, and in the second will be predicted 0.4 dog and 0.5 cat, the accuracy and the loss will both increase. 

"""
# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**

The depth affects the accuracy because it expands the expressiveness of the network, if all calculations are smooth, we should be able to overfit the model
better with a deeper network, but we can see this isn't the case here.
The best results were encountered using a depth of 2.
The networks with depth 4 and 8 were unable to train.
This is because we experience a vanshing gradient; the deepest layers learn faster, and after they reach their minimum they backpropogate
a gradient which can no longer be optimized (a close to zero gradient).
The inputs to the deeper layers are a sum of the ones to the shallow layers, so as we get deeper the weights are much bigger, when backpropogating 
the gradient, the effect on the small weights of the shallow layers is minimal.

In order to eliminate there vanishing gradients, we can try to add batch normalization, to keep all weights similar, or to initialize the weights
with a smaller variance as a function of the number of inputs.

"""

part3_q2 = r"""
**Your answer:**

Here we again see that L8 was not able to train.
We can also see that more filters give us a better accuracy in every iteration. This can be explained because more filters give 
us more learnable parameters, so we are able to learn more from each run, thus we are more accurate in classification.
These extra learnable parameters also show in the test loss, where we can see in the last epochs that the loss is increasing, and the 
slope of increase is bigger for more filters. This is because more filters give us more of a chance to overfit.

"""

part3_q3 = r"""
**Your answer:**

In this experiment we are observing layers with increasing filters per layer. In a network like this, the first layer picks up the rougher features,
and going deeper the layers pick up finer and finer features.
We also notice the number of epochs to train was lower than the first two experiments.
Again, when the network is too deep we are unable to train it, due to similar issues as in experiment 1.1, and the best results are seen in the more shallow network.

"""


part3_q4 = r"""
**Your answer:**


We added a dropout layer and batch normalization. Dropout improves generalization by adding randomness and prevents overfitting.
Batch normalization increases the stability of the calculation and makes it much faster.
We can see that adding these gave us much faster training times.
In the deeper models we are actually able to train as the normalization makes all of our calculations have stable values, and we do not
have vanishing gradients because of this.

"""
# ==============
