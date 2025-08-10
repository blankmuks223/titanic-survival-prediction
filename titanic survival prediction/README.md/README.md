This project predicts passenger survival on the Titanic using machine learning models such as Logistic Regression and Random Forest.
It uses the Titanic dataset for analysis and modeling.

📂 Project Structure
bash
Copy
Edit
titanic-survival-prediction/
├── src/
│   └── titanic_model.py         # Main Python script
├── plots/                       # Output visualizations
│   ├── survival_rate_by_class.png
│   ├── survival_rate_by_sex.png
│   └── feature_importances.png
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
📊 Features
Data Cleaning: Handling missing values, removing unnecessary columns.

Feature Engineering: Adding Title, FamilySize, and AgeGroup.

Exploratory Data Analysis (EDA): Survival rate by gender, class, and other factors.

Modeling: Logistic Regression and Random Forest Classifier.

Visualization: Seaborn and Matplotlib plots showing trends and feature importances.

🚀 How to Run
Clone the repository:

bash
Copy
Edit
git clone https://github.com/yourusername/titanic-survival-prediction.git
cd titanic-survival-prediction
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run the model:

bash
Copy
Edit
python src/titanic_model.py
📈 Results
Logistic Regression: Accuracy ~ X.XX, ROC AUC ~ X.XX

Random Forest: Accuracy ~ X.XX, ROC AUC ~ X.XX

Visualizations are saved in the plots/ directory.