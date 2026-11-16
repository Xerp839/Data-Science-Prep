# Day 1 — Machine Learning Fundamentals (Sun, 15 Nov)

> Your biggest gap, so give it the most energy. Goal for today: explain each concept **in plain words with an example**. Skip the heavy maths. Interviewers at this level want intuition.

**Time split:** Sections 1–3 (morning) · Sections 4–6 (afternoon) · Section 7 + self-test (evening)

**How to read these notes:** Read the explanation first, then use the bullets and tables for revision. For each topic, aim to explain what it does, why it helps, and one example in your own words.

---

## 1. The Big Picture

**Machine Learning** means the computer learns patterns from data instead of us writing rules by hand.

Think of predicting house prices. Instead of manually deciding how much every extra room should cost, we show the model past houses with their features and sale prices. It learns a relationship and uses it to predict prices for unseen houses. The real goal is **generalization**: doing well on new examples, rather than remembering the examples used for training.

| Type | Has labels? | Example | Algorithms |
|---|---|---|---|
| **Supervised** | Yes (we know the answer) | Spam or not spam, house price | Linear/Logistic Regression, Trees, RF, XGBoost, SVM |
| **Unsupervised** | No | Group customers into segments | K-Means, DBSCAN, PCA |
| **Reinforcement** | Reward signal | Game-playing bot, RLHF for LLMs | Q-learning, PPO |

Supervised learning splits into:
- **Regression**: predict a **number** (salary, price, temperature).
- **Classification**: predict a **category** (fraud / not fraud, tumor / no tumor).

### The standard ML workflow (say this when asked "how do you approach an ML problem?")
1. **Understand the business problem.** What decision will this model help make?
2. **Collect and explore the data (EDA).** Distributions, missing values, outliers, class balance.
3. **Clean and preprocess.** Handle missing values, encode categories, scale numbers.
4. **Feature engineering.** Create useful columns (e.g. "days since last purchase").
5. **Split the data.** Train / validation / test (e.g. 70/15/15).
6. **Train a baseline.** Something simple first (logistic regression), then try better models.
7. **Evaluate** with the right metric (not always accuracy!).
8. **Tune hyperparameters** with cross-validation.
9. **Deploy** (API) and **monitor** (drift, performance).

In practice, the business problem decides the rest of the workflow. For fraud detection, catching fraud may matter more than overall accuracy, so that affects the metric and prediction threshold. Split the data before fitting preprocessing steps: the validation and test sets should represent information the model has not learned from.

---

## 2. Core Algorithms (intuition first)

### 2.1 Linear Regression
- **What:** Fits a straight line (or plane) through the data. `price = w1*size + w2*rooms + b`
- **How it learns:** Finds the weights that minimize **MSE** (mean squared error), i.e. the average of (actual − predicted)².
- **Use when:** The relationship is roughly linear and you want an interpretable model.
- **Assumptions (a common question):** linear relationship, independent errors, constant variance of errors (homoscedasticity), little multicollinearity, normally distributed errors.

**Understand it:** A weight tells us how the predicted price changes when one feature increases by one unit, holding the others fixed. If the size coefficient is ₹5,000 per square foot, adding 100 square feet adds ₹5 lakh to the prediction. This describes an association in the fitted model; it does not prove that changing a feature causes the outcome. Squaring errors in MSE makes a large mistake much more costly than a small one.

**Assumption nuance:** Normal errors matter mainly for classical confidence intervals and hypothesis tests; they are not required just to fit a line and make predictions. Strongly correlated features can make individual coefficients unstable even when predictions remain useful.

### 2.2 Logistic Regression
- **What:** A **classification** model despite the name. It computes a linear score, then squashes it through the **sigmoid** to get a probability between 0 and 1.
  `sigmoid(z) = 1 / (1 + e^(-z))`
- **Decision:** probability > 0.5 → class 1 (the threshold can be tuned!).
- **Loss:** Log loss / binary cross-entropy.
- **Why people like it:** fast, interpretable (coefficients show feature impact), a great baseline.
- **Fun link:** your Mind-Scope CNN's last layer was literally `Dense(1, sigmoid)`, i.e. logistic regression on top of CNN features.

**Understand it:** Suppose a spam model outputs 0.8. That is its estimated probability of spam, and a separate threshold turns it into a yes/no decision. Logistic regression learns a linear score in the input features; sigmoid changes the score into a probability but does not make the decision boundary curved. Cross-entropy penalizes a confident wrong prediction strongly, encouraging the model to improve its probability estimates.

### 2.3 Decision Tree
- **What:** A flowchart of yes/no questions. "Is income > 50k? → Is age < 30? → ..."
- **How it splits:** Picks the question that makes the groups as **pure** as possible, measured by **Gini impurity** or **Entropy / Information Gain**.
- **Pros:** easy to explain, handles non-linear data, no scaling needed.
- **Cons:** **overfits easily**. A deep tree memorizes the training data.

**Understand it:** A useful split separates examples with different outcomes. If a group contains equal numbers of fraud and legitimate transactions, it is mixed; a split that creates one mostly-fraud group and one mostly-legitimate group reduces impurity. A tree can keep splitting until it isolates individual training examples, so limits such as maximum depth and minimum samples per leaf help it generalize.

### 2.4 Random Forest (Bagging)
- **What:** Many decision trees, each trained on a **random sample of rows with replacement** (bootstrap), with a **random subset of features considered at each split**. Final answer = **majority vote or averaged class probabilities** (classification), or **average** (regression).
- **Why it works:** Each tree makes different mistakes, and averaging cancels them out. This **reduces variance** (less overfitting).
- **Bonus:** gives **feature importance**.

**Understand it:** Averaging helps most when the trees do not all make the same mistakes. Bootstrap samples and random feature choices make the trees different, so one overconfident tree has less influence on the final answer. This makes a forest more stable than a single deep tree. Feature importance shows what the model used, but it is not proof of causation and can be misleading for correlated features.

### 2.5 Gradient Boosting / XGBoost / LightGBM (Boosting)
- **What:** Trees built **one after another**. Each new tree focuses on fixing the **errors (residuals)** of the previous trees.
- **Why it works:** Step-by-step correction **reduces bias**.
- **XGBoost extras:** regularization (L1/L2), handles missing values, parallel processing, fast.
- **LightGBM:** even faster on big data (grows trees leaf-wise). **CatBoost:** handles categorical features natively.
- **Reality:** for **tabular data**, XGBoost/LightGBM usually wins, even over deep learning.

**Understand it:** Imagine the current model repeatedly underpredicts expensive houses. The next small tree learns a correction for those examples, and we add only a fraction of that correction, controlled by the learning rate. For squared-error regression, the correction targets residuals; more generally, gradient boosting follows the loss's negative gradients. Too many trees or overly strong corrections can fit noise, so validation and early stopping matter.

### ⭐ Bagging vs Boosting (very common question)
| | Bagging (Random Forest) | Boosting (XGBoost) |
|---|---|---|
| Trees built | **In parallel**, independently | **Sequentially**, each fixes the last |
| Main goal | Reduce **variance** | Reduce **bias** |
| Overfitting risk | Low | Higher if not tuned (learning rate, depth) |
| Speed | Easy to parallelize | Slower to train (XGBoost parallelizes inside each tree) |

### 2.6 K-Nearest Neighbors (KNN)
- Classify a point by looking at its **K closest neighbors** and taking a vote. No real "training".
- **Needs feature scaling** (distance-based!). Slow on large data.
- 💡 Link to RAG: vector search is basically **nearest-neighbor search** on embeddings.

**Understand it:** With K=3, a new customer is classified by the majority label among the three nearest training customers. A very small K is sensitive to noise; a very large K can wash out local patterns. Scaling matters because a feature measured in thousands can dominate distances over a feature measured between 0 and 1, even if it is less useful.

### 2.7 Support Vector Machine (SVM)
- Finds the line or boundary with the **maximum margin** between classes.
- **Kernel trick:** handles non-linear data by implicitly mapping it to higher dimensions (RBF kernel).
- Good on small or medium, high-dimensional data (e.g. text before deep learning).

**Understand it:** Several boundaries might separate two classes, but SVM prefers one with the widest gap to the nearest points, called **support vectors**. A wider margin can improve generalization. The hyperparameter **C** controls how strongly training mistakes are penalized: a large C pushes harder to classify training points correctly; a smaller C tolerates more violations for a wider margin.

### 2.8 Naive Bayes
- Uses Bayes' theorem and assumes features are **conditionally independent given the class** ("naive").
- Very fast, works surprisingly well for **text classification / spam**.

**Understand it:** For spam detection, the model combines how common spam is with how likely words such as "prize" and "offer" are within spam versus legitimate messages. The independence assumption treats those words as separate evidence once the class is known. Real words are related, so the assumption is imperfect, but the model can still classify well. Smoothing prevents an unseen word from making a class probability zero.

### 2.9 K-Means (Unsupervised)
1. Pick K random centers.
2. Assign each point to the nearest center.
3. Move each center to the mean of its points.
4. Repeat until nothing changes.
- **How to choose K:** **Elbow method** (plot inertia vs K and look for the "elbow") or **Silhouette score**.
- **Cons:** you must pick K, it assumes round clusters, and it's sensitive to scaling and outliers.

**Understand it:** K-Means tries to minimize squared distances between points and their assigned center. For customer segmentation, centers might represent groups with different spending and visit patterns. These groups have no predefined labels; you inspect them afterward to give them useful names. Different starting centers can give different results, and clusters need not correspond to meaningful business groups.

### 2.10 PCA (Dimensionality Reduction)
- Compresses many features into fewer "principal components" that keep the most **variance** (information).
- Use it for: visualization, speeding up models, removing correlated features.

**Understand it:** If several measurements describe similar things, PCA creates new axes that summarize their shared variation. Each component is a weighted combination of the original features, so PCA is **feature extraction**, not simply choosing existing columns. It does not use the target label: a direction with high variance is not necessarily useful for prediction. Scale features first when their units differ substantially.

### ⭐ When to use what (cheat table)
| Situation | Go-to choice |
|---|---|
| Tabular data, want the best accuracy | XGBoost / LightGBM |
| Need explainability / a baseline | Logistic / Linear Regression, single tree |
| Images | CNN (or a pretrained model like ResNet) |
| Text / language | Transformers / LLMs (classic: TF-IDF + Logistic Regression) |
| No labels, want groups | K-Means / DBSCAN |
| Too many features | PCA, feature selection |

**"Tree-based vs deep learning?"** → Trees for **structured tabular data**: fewer rows needed, faster, interpretable. Deep learning for **unstructured data** (images, text, audio) and huge datasets.

---

## 3. Data Preprocessing & Feature Engineering

### Missing values
- Drop rows or columns (only if very few are missing or the column is mostly empty).
- Impute: **mean/median** for numbers (median if there are outliers), **mode** for categories, or model-based (KNN imputer).
- Add a flag column "was_missing", which is sometimes useful information in itself.

### Categorical encoding
- **One-hot encoding**: one column per category (city_Mumbai, city_Pune). Good for few categories.
- **Label / ordinal encoding**: Low=0, Medium=1, High=2. Use only when there's a natural order.
- **Target encoding**: replace a category with the average target. Good for many categories, but watch for leakage.

**Understand it:** Encoding determines what relationships the model can infer. Giving Mumbai=0, Pune=1, and Delhi=2 may incorrectly suggest an order or numerical distance, so one-hot encoding is often safer for unordered categories. With target encoding, compute training encodings using an out-of-fold approach, and use only training-derived statistics for validation/test data; otherwise, the answer can leak into the feature.

### Feature scaling
- **Standardization (Z-score)**: `(x − mean) / std` → mean 0, std 1.
- **Normalization (Min-Max)**: squeeze to the range [0, 1]. *(You did this in Mind-Scope: pixel / 255.)*
- **Needed for:** KNN, SVM, K-Means, Logistic Regression with regularization, neural networks, PCA.
- **NOT needed for:** tree-based models (they only care about split order).

**Understand it:** Scaling changes the numerical representation, not the meaning of a feature. Fit the mean, standard deviation, minimum, or maximum on the training set, then reuse those values on validation, test, and production inputs. A pipeline helps enforce this inside cross-validation too. For missing values, median imputation is a simple baseline, but a missingness flag can preserve a useful signal such as "income was not reported."

### Outliers
- Detect them with **IQR** (below Q1 − 1.5·IQR or above Q3 + 1.5·IQR), Z-score > 3, or box plots.
- Handle them by removing, capping (winsorizing), log transform, or using robust models (trees).

**Before removing one:** Check whether it is an error or a valid rare event. An impossible age is probably bad data; an unusually large transaction might be exactly the fraud you want to detect. A statistical cutoff is a prompt to investigate, not an automatic reason to delete a row.

### ⚠️ Data Leakage (interviewers love this)
**Leakage = information from the test set or the future sneaks into training**, so the model looks great offline and fails in production.
Examples:
- Scaling or imputing with stats from the **whole dataset before splitting**. Always fit the scaler on train only.
- A feature that's only known *after* the outcome (e.g. "refund_issued" when predicting fraud).
- **Augmenting images before splitting**, so copies of the same image land in both train and test. *(This actually applies to Mind-Scope! See that file.)*
- Time series: randomly shuffling instead of splitting by time.

---

## 4. Model Evaluation Metrics ⭐⭐⭐ (almost guaranteed)

### 4.1 Confusion Matrix (binary classification)

|  | **Predicted Positive** | **Predicted Negative** |
|---|---|---|
| **Actually Positive** | ✅ True Positive (TP) | ❌ False Negative (FN), "missed it" |
| **Actually Negative** | ❌ False Positive (FP), "false alarm" | ✅ True Negative (TN) |

### 4.2 The metrics

| Metric | Formula | Plain English |
|---|---|---|
| **Accuracy** | (TP + TN) / All | % of all predictions that were right |
| **Precision** | TP / (TP + FP) | "Of everything I flagged as positive, how many really were?" |
| **Recall (Sensitivity, TPR)** | TP / (TP + FN) | "Of all the real positives, how many did I catch?" |
| **Specificity (TNR)** | TN / (TN + FP) | "Of all the real negatives, how many did I correctly clear?" |
| **F1 Score** | 2·P·R / (P + R) | Harmonic mean of precision and recall, a balance of both |

### 4.3 When is which more important? (say these examples)
- **Recall matters more** when **missing a positive is dangerous**:
  - Cancer / tumor detection (missing a tumor is terrible; a false alarm just means another test). ← *your Mind-Scope project*
  - Fraud detection (missing fraud costs money).
- **Precision matters more** when **a false alarm is costly**:
  - Spam filter (sending an important email to spam is bad).
  - Recommending a risky investment.
  - An LLM guardrail blocking legit user messages.
- **F1** when you need a balance and the classes are imbalanced.

### 4.4 Why accuracy lies on imbalanced data
If 99% of transactions are legit, a model that **always says "not fraud" is 99% accurate** but catches **zero fraud**. Use precision, recall, F1, PR-AUC or ROC-AUC instead.

### 4.5 ROC-AUC and PR-AUC
- The **ROC curve** plots **TPR (recall) vs FPR** at every threshold from 0 to 1.
- **AUC** = area under that curve. 1.0 is perfect, 0.5 is a coin toss.
  - Intuition: *"the probability that the model ranks a random positive above a random negative."*
- **PR curve** (Precision vs Recall) is **better for heavily imbalanced data** (fraud, rare disease) because ROC can look too optimistic when there are tons of negatives.

**One example connects the metrics:** Suppose 10 of 100 transactions are fraudulent. The model flags 8 transactions, of which 6 are actually fraud. Then TP=6, FP=2, FN=4, and TN=88. Precision is 6/8 = **75%**, recall is 6/10 = **60%**, and accuracy is **94%**. Accuracy looks strong, but the model still misses 4 of the 10 frauds. ROC-AUC measures ranking across thresholds; it does not tell you that predicted probabilities are well calibrated or which operating threshold to use.

### 4.6 The threshold
Models output a probability. 0.5 is just a default.
- Lower the threshold → more positives flagged → recall increases or stays the same; precision often falls.
- Raise it → fewer positives flagged → recall decreases or stays the same; precision often improves.
- Choose the threshold based on the **business cost** of FP vs FN.

**Understand it:** If fraud probabilities of 0.4 are currently treated as negative, lowering the threshold from 0.5 to 0.3 catches them. Some will be real fraud and some false alarms, which explains the usual tradeoff; precision need not change monotonically. Select the threshold on validation data, then assess it on the test set. For a medical model, the acceptable balance also depends on the clinical workflow and the consequences of unnecessary follow-up.

### 4.7 Regression metrics
| Metric | What | Notes |
|---|---|---|
| **MAE** | Average of the absolute errors | Easy to explain ("off by ₹2,000 on average"), robust to outliers |
| **MSE** | Average of the squared errors | Punishes big errors heavily |
| **RMSE** | √MSE | Same unit as the target, punishes big errors |
| **R²** | 1 − (SS_res / SS_tot) | "% of the variance explained". 1 is perfect, 0 is no better than predicting the mean |
| **MAPE** | Mean absolute % error | Business-friendly, breaks when actual values are near 0 |

**Understand it:** Errors of ₹1,000 and ₹9,000 give an MAE of ₹5,000, while RMSE is about ₹6,403 because it emphasizes the larger miss. Choose based on whether occasional large mistakes deserve extra penalty. R² can be negative on unseen data: that means the predictions are worse than using that evaluation set's mean as a constant prediction.

### 4.8 Clustering metrics
Silhouette score (−1 to 1, higher is better) and inertia (within-cluster sum of squares).

---

## 5. Training, Overfitting & Generalization ⭐⭐⭐

### 5.1 Overfitting vs Underfitting
| | Underfitting | Good fit | Overfitting |
|---|---|---|---|
| Train score | Low | High | **Very high** |
| Test score | Low | High | **Much lower** |
| Problem | Model too simple (**high bias**) | — | Model memorized noise (**high variance**) |
| Analogy | Student who didn't study | Understood concepts | Student who memorized last year's paper |

💡 **Your real example:** in Mind-Scope, training accuracy hit **~99.9%** while validation bounced between **~71% and 92%**. That gap is a warning sign of overfitting; a small or unrepresentative validation set can also make scores unstable.

**Understand it:** Training performance tells you how well the model fits examples it has seen. Validation performance estimates how well that learning transfers. If both are poor, investigate underfitting, weak features, or training problems. If training keeps improving while validation loss worsens, extra training is probably teaching the model details that do not generalize.

### 5.2 Bias-Variance Tradeoff
- **Bias** = error from wrong or too-simple assumptions → underfitting.
- **Variance** = error from being too sensitive to the training data → overfitting.
- Making a model more complex lowers bias but raises variance. **The goal is the sweet spot.**
- Total error = Bias² + Variance + Irreducible noise.

### 5.3 How to fix overfitting
- **More data** / **data augmentation** (flip, rotate, zoom images).
- **Simpler model** (fewer layers, smaller max_depth for trees).
- **Regularization** (L1/L2, see below).
- **Dropout** (neural nets): randomly switch off neurons during training.
- **Early stopping**: stop training when validation loss starts going up.
- **Cross-validation** to get a reliable estimate.
- **Ensembles** (Random Forest averages out variance).
- **Feature selection**: remove noisy features.

### How to fix underfitting
A more complex model, more features, less regularization, training longer.

### 5.4 Regularization (L1 vs L2) ⭐
Add a **penalty for large weights** to the loss so the model stays simple.

| | **L1 (Lasso)** | **L2 (Ridge)** |
|---|---|---|
| Penalty | Sum of \|weights\| | Sum of weights² |
| Effect | Can push some weights to **exactly 0** | Shrinks weights **towards** 0 without encouraging sparsity |
| Bonus | **Automatic feature selection** | Handles correlated features well |
| **Elastic Net** | = L1 + L2 mixed | |

Strength is controlled by **λ (lambda) / alpha**. In sklearn's LogisticRegression, `C = 1/λ`, so smaller C means more regularization.

**Understand it:** The objective becomes `prediction loss + lambda × penalty`. The model now has to justify a large weight by improving predictions enough to offset its penalty. L1 can make some coefficients exactly zero, so those features no longer contribute in a linear model. L2 usually keeps features but reduces reliance on any one of them. Too much regularization can cause underfitting, and scaling matters because the penalty acts on coefficient sizes.

**Interview answer:** "L1 performs feature selection because it can set coefficients exactly to zero. L2 generally shrinks coefficients without forcing sparsity, which helps control overfitting."

### 5.5 Train / Validation / Test split
- **Train**: the model learns from this.
- **Validation**: used to tune hyperparameters and pick the best model.
- **Test**: touched **once** at the end for an honest final score.
- **Stratified split**: keeps the same class ratio in each split. Important for imbalanced data.

**Understand it:** Validation is your feedback while developing; the test set is the final exam. Repeatedly changing the model based on test results effectively turns the test set into another validation set. Also split by the real unit of independence: if one patient has multiple scans, keep that patient's scans together so the model is evaluated on new patients.

### 5.6 Cross-Validation
- **K-Fold CV:** split the data into K parts (e.g. 5). Train on 4, test on 1, rotate 5 times, then average the scores.
- Gives a more reliable estimate than one single split, especially with small data.
- **Stratified K-Fold** for imbalanced classification.
- **TimeSeriesSplit** for time data: always train on the past and test on the future. Never shuffle.

**Understand it:** One lucky split can make a model look better than it is. Cross-validation repeats evaluation across different held-out portions, so both the average and spread of scores are useful. Fit preprocessing and resampling separately within each training fold. After choosing hyperparameters, refit on the available development data and evaluate on the untouched test set.

### 5.7 Hyperparameter Tuning
- **Parameters** = learned by the model (weights).
- **Hyperparameters** = set by you before training (learning rate, max_depth, n_estimators, K in KNN).
- Methods:
  - **Grid Search**: try every combination. Thorough but slow.
  - **Random Search**: try random combinations. Often just as good and much faster.
  - **Bayesian Optimization (Optuna)**: smartly picks the next combination based on past results.

### 5.8 Class Imbalance (fraud, churn, disease)
1. **Use the right metric**: precision/recall/F1/PR-AUC, not accuracy.
2. **Class weights**: `class_weight='balanced'` makes mistakes on the rare class cost more. *Usually the first thing to try.*
3. **Resampling:**
   - **Oversampling** the minority (random duplicate, or **SMOTE**, which creates *synthetic* points between existing minority neighbors).
   - **Undersampling** the majority (risk: losing information).
   - ⚠️ Resample **only the training set, after splitting**, or you leak data.
4. **Threshold tuning**: lower the threshold to catch more of the rare class.
5. **Ensembles**: Balanced Random Forest, XGBoost's `scale_pos_weight`.

**Understand it:** Class weights change how costly each training error is; oversampling changes which examples the learner sees more often. Neither creates new real evidence about rare cases. Keep validation and test sets representative of deployment, rather than balancing them to make the scores look better. If using cross-validation, resample only inside each training fold.

---

## 6. Statistics Quick Hits (the DS title means they may ask)

- **Mean vs median:** the median is robust to outliers (salary data → use the median).
- **Standard deviation:** how spread out the data is.
- **Normal distribution:** bell curve. 68% of values within 1 SD, 95% within 2, 99.7% within 3.
- **Correlation vs causation:** ice-cream sales and drowning are correlated (both go up in summer), but one doesn't cause the other.
- **p-value:** assuming the null hypothesis and test assumptions hold, the probability of results at least as extreme as those observed. A value below a prechosen significance level (e.g. 0.05) supports rejecting the null; it is not the probability that the null is true.
- **A/B testing:** split users into control vs variant, compare a metric, and check statistical significance.
- **Type I error** = false positive (rejecting a true null). **Type II error** = false negative.
- **Central Limit Theorem:** under suitable conditions (e.g. independent, identically distributed observations with finite variance), sample means become approximately normally distributed as sample size grows, even when individual values are not normal.

**Understand it:** Statistical significance and practical importance answer different questions. A tiny improvement can be statistically significant with enough data but too small to matter to the business. In an A/B test, choose the main metric and sample-size plan beforehand, randomize fairly, and consider the effect size and confidence interval alongside the p-value.

---

## 7. Model Deployment & Monitoring (classic-ML view)

- **Save the model:** `joblib` / `pickle` (sklearn), `.keras` / SavedModel (TensorFlow), `.pt` (PyTorch), ONNX (portable).
- **Serve it:** wrap it in a **Flask / FastAPI** endpoint `/predict` → **load the model once at startup** (exactly your Presidio fix!), not on every request.
- **Monitor:**
  - **Data drift**: input data distribution changes (e.g. new customer types).
  - **Concept drift**: the relationship between input and output changes (fraud patterns evolve).
  - Track prediction distribution, latency, error rates, and real accuracy once labels arrive.
  - **Retrain** on a schedule or when drift is detected.
- **MLflow**: tracks experiments (params, metrics, model versions). Good to name-drop.

**Understand it:** Deployment must preserve the whole prediction process, including preprocessing and feature order. A model trained on standardized inputs can fail if an API passes raw values. After deployment, labels may arrive late, so input distributions, latency, and error rates provide early signals. Drift is a reason to investigate; it does not automatically prove accuracy has dropped or that immediate retraining will help.

### Mini sklearn template (know how to write this)
```python
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

df = pd.read_csv("data.csv")
X, y = df.drop(columns="churn"), df["churn"]

num_cols = ["age", "monthly_bill"]
cat_cols = ["city", "plan"]

pre = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])

model = Pipeline([
    ("pre", pre),                                      # Pipeline = no leakage: fit only on train
    ("clf", RandomForestClassifier(class_weight="balanced", random_state=42)),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

print(cross_val_score(model, X_train, y_train, cv=5, scoring="f1").mean())
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(classification_report(y_test, pred))
print("ROC-AUC:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
```

---

## ✅ Self-Test (cover the notes and answer OUT LOUD)

1. Difference between regression and classification? Give 2 examples each.
2. Why is logistic regression called regression if it classifies?
3. Bagging vs boosting. Which reduces bias and which reduces variance?
4. Why does XGBoost usually beat neural nets on tabular data?
5. Draw a confusion matrix. Define precision and recall.
6. For tumor detection, which matters more: precision or recall? Why?
7. Why is accuracy bad for fraud detection?
8. What does ROC-AUC = 0.5 mean?
9. What is overfitting? Give 5 ways to fix it.
10. L1 vs L2: which one does feature selection?
11. What is data leakage? Give 2 examples.
12. How do you handle class imbalance? (at least 4 ways)
13. What is K-fold cross-validation and why use it?
14. Which models need feature scaling, and which don't?
15. How do you choose K in K-Means?
16. Grid search vs random search?
17. What's data drift? How would you detect it?
18. Explain the bias-variance tradeoff with an analogy.
