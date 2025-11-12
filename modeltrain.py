import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib 

CSV_FILE_PATH = 'data.csv'

try:
    data = pd.read_csv(CSV_FILE_PATH)
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy tệp {CSV_FILE_PATH}")
    print("Hãy đảm bảo tệp .csv của bạn ở cùng thư mục với script này.")
    exit()

print(f"Đã tải {len(data)} mẫu dữ liệu.")

X = data.drop('label', axis=1)
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y) 

print(f"Bắt đầu huấn luyện trên {len(X_train)} mẫu...")

model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(X_train, y_train)
print("Huấn luyện hoàn tất!")

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Độ chính xác trên bộ kiểm tra: {accuracy * 100:.2f}%")

MODEL_PATH = 'sign_language_model.pkl'
joblib.dump(model, MODEL_PATH)