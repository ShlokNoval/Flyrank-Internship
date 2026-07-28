# The Prompt Ladder: From Lazy to Engineered

**Track:** General AI Fluency (Week 2)  
**Context:** Machine Learning & Data Science Internship (Flyrank)

---

## Baseline: The "Lazy" Prompt
**Prompt:**
> *"Write python code to analyze customer churn data."*

**Simulated Output Excerpt:**
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load generic data
df = pd.read_csv('churn.csv')
X = df.drop('churn', axis=1)
y = df['churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier()
model.fit(X_train, y_train)
print(f"Accuracy: {model.score(X_test, y_test)}")
```

**Notes:**
- **What changed:** N/A (Baseline).
- **What actually improved in the output:** N/A.
- **What still failed:** The output assumes a completely generic dataset format, uses basic methods without context, and just dumps an accuracy score without providing any real business insights. 
- **What to try next:** Add a clearer goal about what kind of insights we need from the data.

---

## Version 1: Clearer Goal
**Prompt:**
> *"Write python code to analyze customer churn data. **Focus specifically on finding which features are most important in predicting whether a high-value customer will leave.***"*

**Simulated Output Excerpt:**
```python
# ... [model training code] ...
importances = model.feature_importances_
for feature, imp in zip(X.columns, importances):
    print(f"{feature}: {imp}")
# (Outputs meaningless column names like Feature_1, Feature_2 since it has no context)
```

**Notes:**
- **What changed:** Added a clearer goal (focusing on feature importance for high-value customers).
- **What actually improved in the output:** The AI stopped just optimizing for raw accuracy and actually wrote code to extract and print feature importances. It shifted from a "predictive" task to an "explanatory" task.
- **What still failed:** The script generated dummy column names (`Feature_1`, `Feature_2`) that make no sense for our actual SaaS business. It requires heavy editing to map to my real database.
- **What to try next:** Provide real context about the dataset schema.

---

## Version 2: Real Context
**Prompt:**
> *"Write python code to analyze customer churn data **for our B2B SaaS platform. The dataset has columns: `mrr`, `login_frequency`, `support_tickets`, `days_since_last_active`, and `churned`.** Focus specifically on finding which features are most important in predicting whether a high-value customer will leave."*

**Simulated Output Excerpt:**
```python
# Filter for high value customers (MRR > 1000)
high_value_df = df[df['mrr'] > 1000]
X = high_value_df[['login_frequency', 'support_tickets', 'days_since_last_active']]
# ... [training code] ...
print("Feature Importances:")
print(f"Days since active: 0.45\nSupport tickets: 0.35\nLogin frequency: 0.20")
```

**Notes:**
- **What changed:** Added real context (business type and dataset schema).
- **What actually improved in the output:** The code is now instantly copy-pasteable. It successfully utilized the `mrr` column to logically define a "high-value customer" and applied my actual database columns directly into the model inputs.
- **What still failed:** The results are printed to the terminal as raw text. If I need to show this to non-technical stakeholders or product managers, command-line printouts are practically useless.
- **What to try next:** Specify an output format.

---

## Version 3: Output Format (The "Made it Worse" Moment)
**Prompt:**
> *"Write python code to analyze customer churn data for our B2B SaaS platform. The dataset has columns: `mrr`, `login_frequency`, `support_tickets`, `days_since_last_active`, and `churned`. Focus specifically on finding which features are most important in predicting whether a high-value customer will leave. **Output the results as a complex interactive D3.js dashboard.***"*

**Simulated Output Excerpt:**
```python
from flask import Flask, render_template, jsonify
# ... 100 lines of complex JSON serialization ...
@app.route('/data')
def get_data():
    return jsonify(feature_importances)
# ... HTML template strings with embedded raw D3.js scripts ...
```

**Notes:**
- **What changed:** Specified an output format (complex interactive D3.js dashboard).
- **What actually improved in the output:** It technically followed instructions and attempted to create visual data representations.
- **What still failed:** This made things *significantly worse*. The code ballooned from 30 lines to over 200 lines, mixed backend Flask routes with frontend Javascript, and failed to run entirely because it required a specific folder structure for HTML templates. The core ML analysis was buried under boilerplate.
- **What to try next:** Pivot the output format to something more practical and add constraints.

---

## Version 4: Constraints
**Prompt:**
> *"Write python code to analyze customer churn data for our B2B SaaS platform. The dataset has columns: `mrr`, `login_frequency`, `support_tickets`, `days_since_last_active`, and `churned`. Focus specifically on finding which features are most important in predicting whether a high-value customer will leave. **Do not build web apps or use HTML/JS. Use only pandas, scikit-learn, and matplotlib/seaborn to output a single cleanly styled PNG chart.***"*

**Simulated Output Excerpt:**
```python
import seaborn as sns
import matplotlib.pyplot as plt
# ... [model code] ...
plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=X.columns, palette='viridis')
plt.title('Top Churn Predictors for High-Value Clients')
plt.savefig('churn_feature_importance.png')
```

**Notes:**
- **What changed:** Added constraints (no web apps/JS, only specific python data libraries, single PNG output).
- **What actually improved in the output:** The web server bloat completely disappeared. The script became hyper-focused, executable in a standard Jupyter notebook environment, and generated exactly the visual asset needed for a presentation.
- **What still failed:** The chart is correct, but Random Forest feature importances only tell you *what* is important, not the *direction* (e.g., do high support tickets increase or decrease churn?). It lacks narrative interpretation.
- **What to try next:** Add quality criteria / review instructions to explain the findings directionally.

---

## Version 5: Quality Criteria
**Prompt:**
> *"Write python code to analyze customer churn data for our B2B SaaS platform. The dataset has columns: `mrr`, `login_frequency`, `support_tickets`, `days_since_last_active`, and `churned`. Focus specifically on finding which features are most important in predicting whether a high-value customer will leave. Do not build web apps or use HTML/JS. Use only pandas, scikit-learn, and matplotlib/seaborn to output a single cleanly styled PNG chart. **Finally, include a function that uses SHAP values to explain the directional impact of the top 3 most important features in plain English.***"*

**Simulated Output Excerpt:**
```python
import shap
# ... [training and plotting code] ...
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
# Function analyzes SHAP correlations
def explain_drivers():
    print("Business Insight Summary:")
    print("1. Days Since Last Active: Higher values strongly increase churn probability.")
    print("2. Support Tickets: High ticket volumes correlate with a moderate increase in churn.")
```

**Notes:**
- **What changed:** Added an explicit quality/verification criteria layer (use SHAP values to explain directional impact in plain English).
- **What actually improved in the output:** The output shifted from "here is a raw chart" to "here is the chart and the actionable business interpretation." The inclusion of SHAP values gave precise, directional insights, bridging the gap between data science and product strategy.
- **What still failed:** Nothing critical. The script is now robust, context-aware, heavily constrained, and provides instantly actionable business insights.
- **What to try next:** N/A - this is a complete prompt.

---

## The Final Reusable Prompt
*Cleaned up and generalized for any Data Scientist on the track to use when doing feature analysis.*

> **Act as a Senior Data Scientist for a B2B SaaS company.**
> 
> **Goal**: Write python code to train a model and identify the top features predicting a specific target outcome.
> 
> **Context**: 
> - The dataset schema is: `[INSERT YOUR COLUMNS HERE]`
> - The target variable to predict is: `[INSERT TARGET VARIABLE HERE]`
> - Sub-population filter (if any): `[e.g., Filter for high MRR customers]`
> 
> **Constraints**:
> - Use only `pandas`, `scikit-learn`, `shap`, and `seaborn`.
> - Do not build web applications or interactive dashboards.
> - Output the feature importance visualization by saving it as a cleanly styled PNG file.
> 
> **Quality Criteria / Verification**:
> - Ensure the script is a single, self-contained Python file.
> - Include a function using SHAP values to automatically print a plain-English explanation of how the top 3 driving features *directionally* impact the target variable.
