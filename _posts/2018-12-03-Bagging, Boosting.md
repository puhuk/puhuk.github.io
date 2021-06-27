---
title: "Bagging, Boosting"
date: 2018-12-06 08:26:28 -0400
categories: ML terms
use_math: true
---

When study Machine Learning, there are some confusing concept which is not easily understand. In my case, I got confused with the terms Bagging and Boosting.
So, let me explain what I studied about Bagging, Boosting and related terms.

#### Description
Bagging and Boosting are both ensamble model which makes prediction with multiple models.
The basic concept of this ensamble is to maximize the prediction rate by combining various models. To acheive this idea into implemetation, tree-based models are widely used as a basic model, because tree model can avoid overfitting occurred when the model complexity increases.
Let's simply compare between logistic and decision tree.
As model increases its complexity, logistic regression ocurrs overfitting as many curves comes up.
For example, imagine the logistic probability
$$
\ln(\frac{\pi(x)}{1-\pi(x)})
$$
while
$$
\pi(x)=\frac{e^{\beta_0+\beta_1X_1+\beta_2X_2+\beta_3X_3+...+\beta_PX_P}}{1+e^{\beta_0+\beta_1X_1+\beta_2X_2+\beta_3X_3+...+\beta_PX_P}}
$$

It makes so many curves to fit the data and eventually will occur overfitting.

However, for decision tree, the prediction depends on the **depth of tree** or **the number of leaf**, not depends on the quantity of data.
And it can prune the tree to avoid the overfitting.

So, basic concept is to enhance the model performance by combining various models which is **1. less influences from overfitting** and **2. little change of model can change the output.** (like decision tree)

There are mainly 2 ways of combining model.
One is Bagging, which is from **B**ootstrap **Agg**regat**ing** and another one is Boosting.

Bagging makes multiple model with subset data sampled with replacement and combines while Boosting combines model from various data which has sampled without replacement.

Random forest is a good example of Bagging algorithm.
This starts from the concept to make better model by combining various decision trees.
To avoid correlation between models, uses random sampled data and to get the best first from each tree, it does not proceed pruning.

So the strong point of Random forest is
1. It is simple even though it includes non-linearity
2. It solves overfitting with baging.

The weekness of Random forest is
1. High computational costs because of non-pruning.
2. As many data has input, has little (no) performance enhancement
   (Actually, it is the specification of decision tree model)

For Boosting, the big difference from Bagging is cut data into multiple sets without replacement.
As proceed training the data, each model corrects the problem from previous models and adapt to each data, which means it corrects and updates from previous models in each stage.

Gradient boosting is one example of Boosting model.
Set the basic model by combining multiple tree models. And combine another model to correct the error from the combination of previous models.

Until here, has deal with Bagging and Boosting.
XGBoost is one of Boosting algorithm and is hot nowadays in many data competition.
In somedays, going to comeback with XGBoost (maybe next or next the next post)

