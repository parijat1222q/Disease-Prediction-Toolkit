# 🩺 Disease Prediction Toolkit

## 📌 Project Description
This project demonstrates how **Machine Learning** can be applied in healthcare to predict heart disease.  
We trained multiple ML models (Logistic Regression, Decision Tree, Random Forest) on the **UCI Heart Disease Dataset**.  
The toolkit includes preprocessing, training, evaluation, visualizations, and predictions for new patient data.  

---

## 📊 Dataset
- **Source**: [UCI Heart Disease Dataset on Kaggle]([https://www.kaggle.com/ronitf/heart-disease-uci](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data))
- **Features**: Age, Blood Pressure, Cholesterol, Max Heart Rate, etc.
- **Target**: Presence of disease (`0 = No`, `1 = Yes`)

---

## ⚙️ Workflow
1. **Preprocessing**  
   - Handle missing values  
   - Encode categorical features  
   - Feature scaling with `StandardScaler`

2. **Model Training**  
   - Logistic Regression  
   - Decision Tree  
   - Random Forest  

3. **Evaluation Metrics**  
   - Accuracy  
   - Precision  
   - Recall  
   - F1 Score  
   - ROC-AUC  

4. **Visualizations**  
   - Confusion Matrix  
   - ROC Curve  
   - Feature Importance  

5. **Prediction**  
   - Supports custom patient input for disease prediction  

---

## 🏆 Results

| Model               | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|----------------------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.85     | 0.82      | 0.87   | 0.84     | 0.86    |
| Decision Tree       | 0.80     | 0.78      | 0.81   | 0.79     | 0.79    |
| Random Forest       | 0.90     | 0.89      | 0.91   | 0.90     | 0.92    |

✅ Best model: **Random Forest** with **92% ROC-AUC**  

---

## 📈 Visualizations
- ✅ Confusion Matrix (Random Forest)  
- ✅ ROC Curve (Random Forest)  
- ✅ Feature Importance (Random Forest)  

---

## ▶️ Demo
🎥 Watch the 30-second demo video here: *(Insert YouTube/Google Drive link after uploading)*

---

## 🚀 How to Run
```bash
# Clone the repo
git clone https://github.com/yourusername/Disease-Prediction-Toolkit.git
cd Disease-Prediction-Toolkit

# Install requirements
pip install -r requirements.txt

# Run the notebook
jupyter notebook Disease_Prediction.ipynb
