import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report, precision_recall_curve
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier
import optuna
from optuna.samplers import TPESampler

def align_test_to_train_columns(df_test: pd.DataFrame, train_columns: list) -> pd.DataFrame:
    if 'Churn' in df_test.columns:
        df_test = df_test.drop('Churn', axis=1)
    df_test_aligned = df_test.reindex(columns=train_columns, fill_value=0)
    return df_test_aligned

def optimize_logistic_regression(trial, X_train, y_train):
    """Оптимизация параметров LogisticRegression"""
    params = {
        'C': trial.suggest_float('lr_C', 0.001, 10, log=True),
        'penalty': trial.suggest_categorical('lr_penalty', ['l1', 'l2']),
        'solver': 'liblinear',  # liblinear работает с L1 и L2
        'class_weight': trial.suggest_categorical('lr_class_weight', ['balanced', None]),
        'max_iter': 1000,
        'random_state': 42
    }
    
    model = LogisticRegression(**params)
    score = cross_val_score(model, X_train, y_train, scoring='f1', cv=3, n_jobs=-1).mean()
    return score

def optimize_catboost(trial, X_train, y_train):
    """Оптимизация параметров CatBoost"""
    params = {
        'iterations': trial.suggest_int('cb_iterations', 500, 3000),
        'learning_rate': trial.suggest_float('cb_learning_rate', 0.001, 0.3, log=True),
        'depth': trial.suggest_int('cb_depth', 3, 10),
        'l2_leaf_reg': trial.suggest_float('cb_l2_leaf_reg', 1, 15),
        'random_strength': trial.suggest_float('cb_random_strength', 0.1, 3.0),
        'bagging_temperature': trial.suggest_float('cb_bagging_temperature', 0.0, 1.0),
        'auto_class_weights': trial.suggest_categorical('cb_class_weight', ['Balanced', None]),
        'early_stopping_rounds': 100,
        'verbose': False,
        'random_state': 42
    }
    
    model = CatBoostClassifier(**params)
    score = cross_val_score(model, X_train, y_train, scoring='f1', cv=3, n_jobs=-1).mean()
    return score

def objective(trial, X_train, y_train):
    """Общая функция оптимизации"""
    # Выбираем какую модель оптимизировать
    model_type = trial.suggest_categorical('model_type', ['logistic', 'catboost'])
    
    if model_type == 'logistic':
        return optimize_logistic_regression(trial, X_train, y_train)
    else:
        return optimize_catboost(trial, X_train, y_train)

if __name__ == '__main__':
    base_dir = '/Users/starfire/Desktop/ALLCHALLENGE/Удержим ли абонента?'
    train_path = f'{base_dir}/train_ohe.csv'
    test_path = f'{base_dir}/testof.csv'
    sample_path = f"{base_dir}/sample_submission (1).csv"

    # Загрузка данных
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    df_sample = pd.read_csv(sample_path)

    X = df_train.drop('Churn', axis=1)
    y = df_train['Churn']

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Оптимизация параметров с Optuna
    print("🔍 Начинаем оптимизацию параметров с Optuna...")
    
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42)
    )
    
    study.optimize(
        lambda trial: objective(trial, X_train, y_train),
        n_trials=50,  # Количество испытаний
        show_progress_bar=True
    )

    print("✅ Оптимизация завершена!")
    print(f"🎯 Лучшее F1 score: {study.best_value:.4f}")
    print("📊 Лучшие параметры:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # Создаем модели с лучшими параметрами
    best_params = study.best_params
    
    if best_params['model_type'] == 'logistic':
        # LogisticRegression с лучшими параметрами
        lr_params = {
            'C': best_params['lr_C'],
            'penalty': best_params['lr_penalty'],
            'solver': 'liblinear',
            'class_weight': best_params['lr_class_weight'],
            'max_iter': 1000,
            'random_state': 42
        }
        lr_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(**lr_params))
        ])
        cb_pipeline = None
        
    else:
        # CatBoost с лучшими параметрами
        cb_params = {
            'iterations': best_params['cb_iterations'],
            'learning_rate': best_params['cb_learning_rate'],
            'depth': best_params['cb_depth'],
            'l2_leaf_reg': best_params['cb_l2_leaf_reg'],
            'random_strength': best_params['cb_random_strength'],
            'bagging_temperature': best_params['cb_bagging_temperature'],
            'auto_class_weights': best_params['cb_class_weight'],
            'early_stopping_rounds': 100,
            'verbose': False,
            'random_state': 42
        }
        lr_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(class_weight='balanced', random_state=42))
        ])
        cb_pipeline = Pipeline([
            ('clf', CatBoostClassifier(**cb_params))
        ])

    # Обучение и оценка
    models = {}
    
    if best_params['model_type'] == 'logistic':
        print("\n--- Обучение лучшей LogisticRegression ---")
        lr_pipeline.fit(X_train, y_train)
        y_val_pred = lr_pipeline.predict(X_val)
        score = f1_score(y_val, y_val_pred)
        print(f'F1 на валидации: {score:.4f}')
        best_model = lr_pipeline
        best_model_name = 'LogisticRegression'
        
    else:
        print("\n--- Обучение моделей ---")
        # Обучаем обе модели для ансамбля
        lr_pipeline.fit(X_train, y_train)
        cb_pipeline.fit(X_train, y_train)
        
        # Создаем ансамбль
        ensemble = VotingClassifier(
            estimators=[
                ('lr', lr_pipeline.named_steps['clf']),
                ('catboost', cb_pipeline.named_steps['clf'])
            ],
            voting='soft',
            n_jobs=-1
        )
        ensemble.fit(X_train, y_train)
        
        # Оценка всех моделей
        y_val_pred_lr = lr_pipeline.predict(X_val)
        y_val_pred_cb = cb_pipeline.predict(X_val)
        y_val_pred_ensemble = ensemble.predict(X_val)
        
        score_lr = f1_score(y_val, y_val_pred_lr)
        score_cb = f1_score(y_val, y_val_pred_cb)
        score_ensemble = f1_score(y_val, y_val_pred_ensemble)
        
        print(f'LogisticRegression F1: {score_lr:.4f}')
        print(f'CatBoost F1: {score_cb:.4f}')
        print(f'Ансамбль F1: {score_ensemble:.4f}')
        
        # Выбираем лучшую модель
        best_score = max(score_lr, score_cb, score_ensemble)
        if best_score == score_ensemble:
            best_model = ensemble
            best_model_name = 'Ensemble'
        elif best_score == score_cb:
            best_model = cb_pipeline
            best_model_name = 'CatBoost'
        else:
            best_model = lr_pipeline
            best_model_name = 'LogisticRegression'

    # Подготовка тестовых данных
    if 'id' in df_test.columns and 'id' in df_sample.columns:
        df_test_ordered = df_test.set_index('id').loc[df_sample['id']].reset_index()
    else:
        df_test_ordered = df_test

    features_test = df_test_ordered.drop(columns=[c for c in ['id', 'Churn'] if c in df_test_ordered.columns])
    df_test_aligned = align_test_to_train_columns(features_test, X.columns.tolist())

    # Предсказание на тесте
    test_proba = best_model.predict_proba(df_test_aligned)[:, 1]
    
    # Подбор оптимального порога
    y_val_proba = best_model.predict_proba(X_val)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_val, y_val_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_threshold = thresholds[np.argmax(f1_scores)]
    best_f1 = np.max(f1_scores)
    
    print(f"🎯 Лучший порог: {best_threshold:.3f}, F1: {best_f1:.4f}")
    
    # Применяем порог
    test_labels = (test_proba >= best_threshold).astype(int)

    # Сохранение результатов
    label_map = {1: 'Yes', 0: 'No'}
    out = pd.DataFrame({
        'id': df_sample['id'],
        'Churn': [label_map[v] for v in test_labels]
    })
    
    out_path = f"{base_dir}/submission_optuna_{best_model_name}_f1_{best_f1:.4f}.csv"
    out.to_csv(out_path, index=False)
    print(f'💾 Результаты сохранены: {out_path}')

    # Визуализация процесса оптимизации
    try:
        optuna.visualization.plot_optimization_history(study).show()
        optuna.visualization.plot_param_importances(study).show()
    except:
        print("📊 Визуализация доступна в Jupyter Notebook")