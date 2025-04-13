import pandas as pd
import ipaddress
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
import joblib
from imblearn.over_sampling import SMOTE
from preprocessing import preprocess_pcapng

# CSV 파일을 로드하여 DataFrame으로 변환
df = pd.read_csv('Slowloris1 packets.csv')  #파일 변경하기

# IP 주소 변환 확인
df['Source_IP'] = df['Source'].apply(lambda x: int(ipaddress.IPv4Address((x))))
df['Destination_IP'] = df['Destination'].apply(lambda x: int(ipaddress.IPv4Address((x))))

print(df[['Source_IP', 'Destination_IP']].head())

# 숫자형 데이터만 추출하여 NaN 값을 처리
numeric_columns = df.select_dtypes(include=['number']).columns
imputer = SimpleImputer(strategy='mean')
df[numeric_columns] = imputer.fit_transform(df[numeric_columns])

# 2. 모델 훈련
X = df.drop(columns=['Label', 'Source', 'Destination'])  # 'Label' 컬럼은 실제 공격 유형을 나타냅니다.
y = df['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SMOTE를 사용하여 데이터 불균형 문제 해결
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

scaler = StandardScaler()
X_resampled = scaler.fit_transform(X_resampled)
X_test = scaler.transform(X_test)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_resampled, y_resampled)
joblib.dump(model, 'dos_attack_classifier.pkl')
joblib.dump(scaler, 'scaler.pkl')

# 3. 모델 평가
model = joblib.load('dos_attack_classifier.pkl')
scaler = joblib.load('scaler.pkl')
X_test = scaler.transform(X_test)
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, zero_division=0))
