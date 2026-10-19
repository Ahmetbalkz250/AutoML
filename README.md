# AutoML Studio Pro 🚀

AutoML Studio Pro is a professional desktop application built with Python and `CustomTkinter` that allows users to train Machine Learning models without writing a single line of code. With its user-friendly interface, you can seamlessly load datasets, select target columns, and instantly view model performances with advanced analysis metrics.

## 🌟 Advanced Features
- **Modular Architecture (OOP):** The project is cleanly separated into GUI (`main_gui.py`), Data Processing (`data_processor.py`), and ML Engines (`models_classification.py`, `models_regression.py`). It provides a scalable and maintainable structure, far away from spaghetti code.
- **5 Core Algorithms:** Performs Classification (Logistic Regression, SVC, Gaussian Naive Bayes) and Regression (Linear Regression, SVR) tasks with a single click.
- **Auto-Scaling:** Automatically aligns all data with `StandardScaler` prior to training to prevent mathematical crashes in distance-based models (SVC/SVR).
- **Comprehensive Model Analysis (NEW):** Evaluates models using a true data scientist approach rather than a single score, providing detailed reports:
  - **For Classification:** `Weighted F1-Score` for imbalanced datasets, `AUC (Area Under the Curve)` for probability separation success, and `Confusion Matrix` for prediction details.
  - **For Regression:** `R2-Score` for the model's data explanation variance, along with `MAE (Mean Absolute Error)` and `MSE (Mean Squared Error)` to observe the average deviation.
- **Smart Data Filtering:** Automatically detects and drops empty cells (`NaN`) to ensure system stability.

## 📖 How to Use
Using the application is extremely simple:
1. **Load Data:** Click the *"📂 Select Data File (.csv / .xlsx)"* button on the interface and choose the dataset on your computer. The system will read the file and detect column headers in milliseconds.
2. **Set Target:** Select the **Target Column** you want to predict from the dropdown menu.
3. **Ignite the Model:** Click one of the 5 buttons at the bottom depending on your dataset type:
   * If your target is a category (e.g., Sick/Healthy, Spam/Not Spam), use the **Blue (Classification)** buttons.
   * If your target is a numerical value (e.g., House Price, Temperature), use the **Orange (Regression)** buttons.
4. **Review Results:** The model will be trained in seconds, and a stylish popup report containing advanced metrics like AUC, MAE, R2, and Confusion Matrix will appear on your screen.

## 🛠️ Data Requirements (MVP Version V1.0)
This version has a strict and secure data acceptance mechanism to reduce the mathematical error margin to zero. To get the best results, your dataset must comply with the following rules:
1. **Format:** Only `.csv` and `.xlsx` file extensions are supported.
2. **Numeric Data Only:** The dataset must **not** contain text (string) based categorical variables such as "Yes/No" or "Male/Female". Users must convert these texts to numbers (Label Encoding) in their Excel/CSV files before loading them into the application. To prevent system crashes, the engine automatically ignores columns containing text.
3. **Clean Headers:** Column names must not have invisible trailing or leading spaces (e.g., `"Price "` should be `"Price"`).

## 🚀 Installation and Execution
1. After downloading the project to your local machine, install the required libraries:
```bash
pip install pandas scikit-learn customtkinter openpyxl
```

2. Run the main interface file:
```bash
python main_gui.py
```