import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix

warnings.filterwarnings('ignore')

def train_logistic_regression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logistic_model = LogisticRegression(max_iter=1000)
    logistic_model.fit(X_train_scaled, y_train)
    y_pred = logistic_model.predict(X_test_scaled)

    # METRİKLERİ HESAPLAMA
    f1 = f1_score(y_test, y_pred, average='weighted') 
    cm = confusion_matrix(y_test, y_pred)
    
    try:
        y_prob = logistic_model.predict_proba(X_test_scaled)
        if len(logistic_model.classes_) == 2: # İki sınıflıysa
            auc = roc_auc_score(y_test, y_prob[:, 1])
        else: # Çok sınıflıysa
            auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
        auc = round(auc, 4)
    except:
        auc = "Hesaplanamadı"

    # Tek skor yerine tüm metrikleri paketleyip gönderiyoruz
    metrikler = {"F1": f1, "AUC": auc, "CM": cm}
    return metrikler, logistic_model, scaler


def train_svc(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    svc = SVC(kernel='rbf', probability=True)
    svc.fit(X_train_scaled, y_train)
    y_pred = svc.predict(X_test_scaled)
    
    # METRİKLERİ HESAPLAMA
    f1 = f1_score(y_test, y_pred, average='weighted') 
    cm = confusion_matrix(y_test, y_pred)
    
    try:
        y_prob = svc.predict_proba(X_test_scaled)
        if len(svc.classes_) == 2:
            auc = roc_auc_score(y_test, y_prob[:, 1])
        else:
            auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
        auc = round(auc, 4)
    except:
        auc = "Hesaplanamadı"

    metrikler = {"F1": f1, "AUC": auc, "CM": cm}
    return metrikler, svc, scaler

def train_gaussienbayes(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    gnb = GaussianNB()
    gnb.fit(X_train_scaled, y_train)
    y_pred = gnb.predict(X_test_scaled)
    f1 = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)
    
    try:
        y_prob = gnb.predict_proba(X_test_scaled)
        if len(gnb.classes_) == 2:
            auc = roc_auc_score(y_test, y_prob[:, 1])
        else:
            auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
        auc = round(auc, 4)
    except:
        auc = "Hesaplanamadı"

    metrikler = {"F1": f1, "AUC": auc, "CM": cm}
    return metrikler, gnb, scaler

