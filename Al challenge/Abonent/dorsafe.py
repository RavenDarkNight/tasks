import pandas as pd

# Загрузка
df = pd.read_csv("/Users/starfire/Desktop/ALLCHALLENGE/test (1).csv")
numeric_candidates = ["tenure", "MonthlyCharges", "TotalCharges"]
for col in numeric_candidates:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
# Пропуски в числовых столбцах заполним нулями (или можно median)
for col in numeric_candidates:
    if col in df.columns:
        median_value = df[col].median()
        df[col]= df[col].fillna(median_value)


# Приведём сервисные категории: заменим "No internet service"/"No phone service" -> "No"
service = [
    'MultipleLines',
    'OnlineSecurity',
    'OnlineBackup',
    'DeviceProtection',
    'TechSupport',
    'StreamingTV',
    'StreamingMovies',
]

for col in service:
    if col in df.columns:
        df[col] = df[col].replace({
            "No internet service": "No",
            "No phone service": "No",
        })

# Бинарные столбцы Yes/No -> 1/0
binar_col = [
    'Partner',
    'Dependents',
    'PhoneService',
    'PaperlessBilling',
]

for col in binar_col:
    if col in df.columns:
        df[col] = df[col].map({"Yes": 1, "No": 0}).astype("Int32")

if "gender" in df.columns:
    df["gender"] = df["gender"].map({"Male": 1, "Female": 0}).astype("Int32")

target_col = 'Churn'
if target_col in df.columns:
    df[target_col] = df[target_col].map({'Yes':1, 'No':0}).astype('Int32')

known_categorical = [
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaymentMethod",
]

# Исключим бинарные, которые уже перевели в 0/1, и таргет
exclude_from_dummies = set(binar_col + [target_col, "gender"])  # gender уже переведён в 0/1
columns_for_dummies = [
    c for c in known_categorical
    if (c in df.columns) and (c not in exclude_from_dummies)
]

# One-Hot кодировка оставшихся категориальных
df_ohe = pd.get_dummies(df, columns=columns_for_dummies, drop_first=False)

# Сохранить результат
output_path = "/Users/starfire/Desktop/ALLCHALLENGE/testof.csv"
df_ohe.to_csv(output_path, index=False)

print("Сохранено:", output_path)
print("Форма данных:", df_ohe.shape)