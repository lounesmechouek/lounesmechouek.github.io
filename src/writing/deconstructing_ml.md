---
title: "Deconstructing Machine Learning: The Core Technical Components"
date: 2024-03-13
category: "Concept Explainers"
excerpt: "A practical breakdown of the four pillars of machine learning, with concrete task examples and the tradeoffs behind each component."
coverImage: "/assets/images/writing/medium_datapipeline.png"
---

Dive into the core of machine learning and uncover the fundamental pillars that orchestrate its transformative power.

Welcome to the world of machine learning, where raw data transforms into knowledge, models predict the future, and algorithms optimize performance. In this article, we explore the essential components of machine learning, unpacking them with detailed explanations and concrete examples.

> N. B.: This article primarily focuses on the fundamental technical components enabling the learning and functioning of machine learning models. Broader concepts related to machine learning systems (or products), such as deployment, monitoring, and MLOps, are beyond the scope of this article and will be explored in future publications.

## The Four Pillars of Machine Learning

At the heart of what we commonly refer to as "AI" lies a subset known as machine learning. This domain looks into crafting algorithms and techniques that empower computers to execute tasks without explicit programming instructions.

In order to enable this, machine learning techniques are typically defined according to a well-known structure containing four main pillars. These pillars allow for the complete characterization of the nature of the problem to be solved, as well as the solution provided for this purpose. These four components are: the data, the model, the objective, and the optimizer.

### Data: The Lifeblood of ML

Let us start with the foundation: data. Think of it as the raw material that powers machine learning. It is the information fed into the models to teach them about the reality of the task in hand.

This data, characterized by its diverse origins, types, formats, and representations, defines the tasks and their handling methods. It can originate from multiple sources like databases, files and real-time streams. It can take various forms, such as quantitative or qualitative data, discrete or continuous values, and exist in different formats, including tabular, text, images, videos, and sounds. Additionally, it can represent different concepts, like time series, ratings, and even relationships and networks.

Beyond its characteristics, data in machine learning hinges on another crucial concept: annotation (labeling). Imagine you are building a system to predict patient illnesses. Each record needs a label, like "sick" or "non-sick". This time-consuming process, often requiring domain experts (doctors in this case), determines the data quality and quantity, impacting the choice of supervised, unsupervised, or other machine learning models that we will examine in the next section.

For all these reasons, the journey from raw data to actionable insights is far from trivial. It demands a good understanding of the data (particularly through visualization), meticulous preprocessing, feature engineering, and sometimes domain expertise to extract meaningful signals from the noise during the modeling step.

### Model: The Mathematical Representation

At its core, a machine learning model is an approximation of a function F that best describes the relationships between the features of a dataset. The model uses parameters theta (and potentially other latent variables Z) that it gradually improves during the learning process to make this approximation. It can then generalize this description to new, unseen data, albeit with a certain error rate.

Building or selecting an appropriate ML model is a lengthy iterative process that entails numerous steps. The primary factor to consider is data. The number of available labels (expected outputs) compared to the number of observations in the dataset will first define the preferred algorithm paradigm. In the best-case scenario where there is sufficient labeled data, supervised models will generally be preferred to take advantage of this information. Conversely, if there are no annotations, unsupervised or self-supervised methods will be preferred. In between, there are semi-supervised approaches that combine the best of both worlds. Finally, the type of expected outputs will also define the nature of the model: regression or classification.

A number of important concepts can help distinguish between suitable models after several trial-and-error processes. These include, among others: the performance evaluated on test and validation datasets, and the complexity of each model relative to the business needs.

Once the model has been selected, it is necessary to define its parameters. This is where two key concepts of ML come into play: the objective function and the optimization algorithm. These concepts allow the machine to learn to perform a task (define the optimal parameters) without being explicitly programmed.

### Objective: The Guiding Light

The objective function (or criterion) plays a crucial role during the process of training a machine learning model by guiding the adjustment of parameter values at each iteration. In other words, the overarching aim of the training process is to find the set of parameters that minimizes (or in some cases, maximizes) this function, thereby optimizing the model's performance for the given task.

> Although a subtle difference may exist between the terms objective function, cost function, and loss function, in this article we use these three terminologies interchangeably.

The choice of objective function depends on the specific problem and the desired outcome. For instance, in regression tasks, such as predicting house prices based on features like size and location, the mean squared error (MSE) between the predicted prices and the actual prices in the training dataset can be used as an objective function. On the other hand, in classification tasks, cross-entropy measures the discrepancy between the model's predicted probability distribution and the actual distribution of the target variable.

### Optimizer: The Path to Perfection

Finally, the last component that facilitates the training process in machine learning is the optimization algorithm. Tasked with the objective to be achieved, the optimizer aids the model in adjusting its parameters to approximate the global optimum of the objective function.

Various optimization algorithms have their own strengths and weaknesses, and the choice depends on factors like the complexity of the model and the size of the dataset. Examples of well-known optimizer families include gradient descent and its variants (SGD, mini-batch GD, etc.), as well as expectation-maximization (EM) and its variants (CEM, GEM, SEM, etc.).

Now that we have discussed the fundamental components of a machine learning system, it is important to mention that not every problem is best solved with machine learning. For instance, when the problem to be solved is not sufficiently complex (few descriptors) or when it is related to a given statistical population whose sample size is too small (few observations), ML methods can be less effective from a performance-to-complexity standpoint.

## Concrete Examples

In this chapter, we dig deeper into how machine learning tackles real-world challenges by exploring several common tasks, including document clustering, object detection, and sales price prediction. For each task, we dissect the inner workings of a representative machine learning system and its key fundamental components (data, model, objective function, optimizer).

### Document Clustering

While document classification assigns predefined labels to documents, document clustering takes a different approach. Here, the goal is to automatically group similar documents together based solely on their content. Imagine a vast archive of uncategorized news articles. Document clustering can analyze the content of each article, identifying keywords, themes, or writing styles, and then group them into meaningful categories, like business, sports, or entertainment, without any prior labeling.

**Data:** The data takes the form of a corpus, a collection of text documents. Before the machine learning model can analyze this data, it needs some preparation using well-established NLP techniques: tokenization, stemming or lemmatization, stop-word removal, and vectorization (BoW, tf-idf, embeddings, etc.). Once preprocessed, the data becomes a collection of continuous numeric vectors representing the documents in a mathematical format, and the task is assigning a cluster number for each vector (document).

**Model:** One widely used algorithm for document clustering is k-means. It operates in two main phases:

1. **Initialization:** Randomly select K elements from the corpus as initial centroids. These centroids represent the centers of the clusters that will be formed.
2. **Iteration:** Repeat two steps until clusters stabilize. First, assign each element to the closest centroid based on a distance measure. Then update centroids to be the average of the elements assigned to them. The key nuance when working with text embeddings is that the data is directional. Proximity is better gauged with cosine or Jaccard similarity, applied to normalized vector representations. In this context, we can use spherical k-means with cosine similarity.

**Objective:** In standard k-means, the algorithm minimizes the intra-cluster inertia criterion. In spherical k-means, we aim to minimize the angular dispersion (or variance of angles) of the documents around their cluster centers.

**Optimizer:** A mixture of von Mises-Fisher distributions can approximate the distribution of normalized document embeddings. Under this assumption, optimizing the classification log-likelihood of the distribution is equivalent to optimizing the angular dispersion criterion of spherical k-means under two conditions: (1) the concentration parameter kappa is constant, (2) all clusters have the same proportion. Therefore, spherical k-means can be viewed as a special case of applying the classification expectation-maximization (CEM) algorithm to estimate the parameters of the mixture distribution under those conditions.

### Object Detection

Imagine a self-driving car navigating a busy street. To function safely, the car needs to not only understand the road layout but also identify and locate objects around it: pedestrians, vehicles, traffic signals, and more. This is precisely where object detection comes in.

**Data:** In the context of object detection, the data used to train the model are images. Each image is represented by three numeric matrices of dimension N x M, where N is the height of the image in pixels and M is the width. Each matrix represents the intensity values of the image pixels for one of the primary colors: red, green, and blue. Before the modeling stage, these matrices are typically normalized to obtain a consistent measurement scale.

**Model:** In the vast landscape of object detection models, one particularly stood out for its simplicity and efficiency: You Only Look Once (YOLO). At the time, this groundbreaking model revolutionized the field by performing object detection in a single pass, as opposed to the multi-stage approaches used by previous models like R-CNN.

How YOLO (v1) works:

1. The input image is resized to a fixed size (typically 3 channels x 416 height x 416 width)
2. The image is divided into a grid of SxS cells
3. For each cell, B anchor boxes are generated to represent different possible object sizes and aspect ratios at that location
4. A convolutional neural network (CNN) backbone extracts features from the image
5. For each cell, the model predicts B vectors of dimension 5+C. The first 5 elements represent objectness, center, width, and height. The remaining C elements represent class probabilities
6. Non-maximal suppression removes duplicate detections

**Objective:** The YOLO loss function is a combination of several terms that penalize the model for different types of errors:

1. **Localization error:** squared error on the center coordinates and on the height and width of each bounding box
2. **Confidence error:** penalizes incorrect prediction of object presence in a cell
3. **Classification error:** measures how well the model distinguishes between classes

**Optimizer:** The optimization of the YOLO objective function is typically performed using a gradient descent variant such as momentum SGD or Adam.

### Sales Price Prediction

In the fast-paced world of business, accurately predicting the selling price of a product or service can be a game-changer. For instance, in the real estate industry, sales price prediction for houses is a crucial task enabling accurate estimations of property values based on attributes such as location, area, and more. This is where ML comes into play.

**Data:** The dataset for this task is tabular, comprising multiple features describing each property, including geographical location, square footage, number of bedrooms and bathrooms, proximity to schools or parks, and other relevant factors. Data preprocessing involves handling missing values, encoding categorical variables, and scaling numerical features to ensure optimal model performance.

**Model:** Extreme Gradient Boosting (XGBoost), a scalable ensemble learning algorithm, stands out as a powerful tool for this task. It fits a collection of weak learners, typically decision trees, to iteratively improve predictive accuracy (or reduce the error) of the previous tree.

**Objective:** XGBoost offers flexibility in choosing the objective function. Differentiable regression loss functions such as MSE can be used with regularization terms to prevent overfitting.

**Optimizer:** XGBoost utilizes gradient boosting for its core optimization. It computes gradients of the loss function with respect to the predictions at each step t, representing the errors or pseudo-residuals. These gradients guide the construction of a new decision tree in a greedy fashion, aiming to minimize the loss function. The new tree is then added to the ensemble, typically by applying a learning rate to control its contribution. This process is repeated for a specified number of trees T, which is a hyperparameter specified by the user. The global procedure resembles gradient descent, albeit instead of updating model parameters directly, XGBoost updates the structure of the ensemble by adding trees.

In our exploration of the fundamental pillars of machine learning systems, we have uncovered the intricate interplay between data, models, objectives and optimizers, guiding the way from raw data to actionable insights. However, not all problems have the luxury of abundant data. In a future article, we will focus on the techniques for building effective ML models even when faced with limited observations or scarce labels. We will explore strategies for data augmentation, leveraging unlabeled data, and selecting appropriate models for these scenarios.

Join the conversation: Can you think of an example from your daily life where machine learning is used? Can you identify the four pillars at play in that example? Share your thoughts.
