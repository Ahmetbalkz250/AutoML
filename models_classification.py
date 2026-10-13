import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import f1_score

warnings.filterwarnings('ignore')

def train_logistic_regression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logistic_model = LogisticRegression(max_iter=1000)
    logistic_model.fit(X_train_scaled, y_train)
    y_pred = logistic_model.predict(X_test_scaled)

    skor = f1_score(y_test, y_pred, average='weighted') 
    
    return skor, logistic_model, scaler

def train_svc(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    svc = SVC(kernel='rbf', probability=True)
    svc.fit(X_train_scaled, y_train)
    y_pred = svc.predict(X_test_scaled)
    score = f1_score(y_test, y_pred, average='weighted') 

    return score, svc, scaler
