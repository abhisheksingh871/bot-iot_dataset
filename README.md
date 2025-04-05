# 🛡️ Bot-IoT Network Attack Detection using Random Forest

This project uses the **Bot-IoT dataset** to build a machine learning model that detects network attacks in IoT environments. 
The model is trained using a **Random Forest classifier**, followed by hyperparameter tuning using **Grid Search** and **Bayesian Optimization** to improve performance.

---

## 📂 Project Structure

---

## 📊 Dataset: Bot-IoT

The **Bot-IoT dataset** is designed for detecting various types of IoT-based network attacks (e.g., DDoS, DoS, reconnaissance). It includes both normal and malicious network traffic, extracted features such as:

- Flow duration
- Source/Destination IPs and ports
- Protocols
- Packet sizes and counts
- Timestamps
- Flags

---

## 🧠 Model: Random Forest Classifier

The model uses a **Random Forest Classifier** as the base algorithm due to its robustness and ability to handle high-dimensional data. The pipeline includes:

- Data Preprocessing:
  - Label encoding / One-hot encoding for categorical features
  - Standard scaling for numerical features
- Train-test split
- Model training with default hyperparameters
- Evaluation using:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - ROC-AUC

---

## ⚙️ Hyperparameter Tuning

Two approaches were used to optimize the model:

### 🔍 Grid Search

- Performed exhaustive search over specified parameter combinations:
  - `n_estimators`
  - `max_depth`
  - `min_samples_split`
  - `min_samples_leaf`

### 🤖 Bayesian Optimization

- Used `BayesSearchCV` from **scikit-optimize** to intelligently search the hyperparameter space using probabilistic modeling.

---

> Final model selected based on best accuracy: ✅ *Bayesian/ Grid Search* (based on results)

---


## 🧠 Model Interpretation with SHAP

To enhance interpretability:

- SHAP (SHapley Additive exPlanations) was used to explain feature contributions.
- Summary plot and force plot were generated for insight into individual predictions.

---

## 💾 Model Saving

The best model was saved using `joblib`:

```python

--------------
📚 References
Bot-IoT Dataset
Moustafa, Nour, and Jill Slay. "The evaluation of network anomaly detection systems: Statistical analysis of the UNSW-NB15 data set and the comparison with the KDD99 data set." Information Security Journal: A Global Perspective 25.1-3 (2016): 18-31.
🔗 UNSW Canberra Cyber Bot-IoT Dataset

Random Forest Classifier - scikit-learn
🔗 https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

Grid Search CV - scikit-learn
🔗 https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

Bayesian Optimization with BayesSearchCV - scikit-optimize
🔗 https://scikit-optimize.github.io/stable/modules/generated/skopt.BayesSearchCV.html

SHAP (SHapley Additive exPlanations)
Lundberg, Scott M., and Su-In Lee. "A unified approach to interpreting model predictions." Advances in neural information processing systems 30 (2017).
🔗 https://shap.readthedocs.io/en/latest/

Joblib (for model serialization)
🔗 https://joblib.readthedocs.io/en/latest/
joblib.dump(final_model, 'final_random_forest_model.pkl')
