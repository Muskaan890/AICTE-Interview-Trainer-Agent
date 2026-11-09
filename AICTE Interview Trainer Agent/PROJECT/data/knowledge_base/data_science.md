# Data Science & Machine Learning Interview Guide

## Core Knowledge Areas

### Machine Learning Fundamentals
Candidates must demonstrate understanding of:
- Supervised Learning: Regression (Linear, Ridge, Lasso), Classification (Logistic Regression, SVM, Decision Trees, Random Forest, XGBoost)
- Unsupervised Learning: K-Means, DBSCAN, PCA, t-SNE, Hierarchical Clustering
- Reinforcement Learning: Q-Learning, Policy Gradient methods
- Deep Learning: CNNs, RNNs, LSTMs, Transformers, Attention Mechanism

### Model Evaluation & Validation
- Train/Validation/Test split rationale
- Cross-validation (k-fold, stratified)
- Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC, RMSE, MAE
- Bias-Variance tradeoff
- Overfitting vs Underfitting: detection and remedies
- Hyperparameter tuning: Grid Search, Random Search, Bayesian Optimization

### Statistics & Probability
- Hypothesis testing (t-test, chi-square, ANOVA)
- Probability distributions (Normal, Binomial, Poisson)
- Bayes' Theorem and Naive Bayes classifier
- Central Limit Theorem
- Confidence intervals and p-values
- A/B testing methodology

### Feature Engineering
- Handling missing values: imputation strategies
- Feature scaling: StandardScaler, MinMaxScaler, RobustScaler
- Encoding categorical variables: One-Hot, Label, Target Encoding
- Feature selection: correlation analysis, mutual information, LASSO
- Dimensionality reduction: PCA, SVD, t-SNE, UMAP

### Deep Learning & LLMs (2024+)
- Transformer architecture and self-attention
- Pre-trained models: BERT, GPT, IBM Granite, LLaMA
- Fine-tuning vs Prompt Engineering vs RAG
- Retrieval-Augmented Generation (RAG) for domain-specific AI
- Vector databases: ChromaDB, Pinecone, Weaviate, IBM Watson Discovery
- IBM watsonx.ai: enterprise AI platform for model development and governance

## Common Interview Questions — Data Scientist

1. Explain the bias-variance tradeoff with an example.
2. How would you handle class imbalance in a fraud detection model?
3. Walk me through your feature engineering process for a tabular dataset.
4. Explain how Random Forest works. Why is it better than a single Decision Tree?
5. What is the difference between bagging and boosting?
6. How do you interpret the coefficients of a Logistic Regression?
7. Describe a model you built from scratch. What was the business impact?
8. How would you deploy an ML model to production? What challenges arise?
9. Explain gradient descent and its variants (SGD, Adam, RMSProp).
10. What is regularization? Explain L1 vs L2 regularization.

## IBM-Specific AI/ML Questions
- How does IBM Granite differ from other foundation models?
- Explain IBM's Responsible AI principles (transparency, explainability, fairness).
- What is Watson Studio and how is it used for ML workflows?
- Describe how watsonx.governance helps manage AI model risk.

## Tools & Technologies Expected
- **Languages**: Python (primary), R (statistical), SQL (essential)
- **ML Frameworks**: TensorFlow, PyTorch, scikit-learn, XGBoost
- **Data Processing**: Pandas, NumPy, PySpark, Dask
- **Visualization**: Matplotlib, Seaborn, Plotly, Tableau
- **Cloud**: IBM Cloud / Watson, AWS SageMaker, Azure ML, GCP Vertex AI
- **MLOps**: MLflow, Kubeflow, IBM Watson Studio pipelines

## Experience Level Expectations — Data Scientist
- **Fresher**: Python, basic ML algorithms, data cleaning, Pandas/NumPy
- **Junior**: End-to-end ML project, feature engineering, model evaluation
- **Mid-Level**: Production model deployment, A/B testing, performance monitoring
- **Senior**: Research-driven improvements, system architecture, stakeholder communication
- **Lead**: Team mentoring, AI strategy, responsible AI governance
