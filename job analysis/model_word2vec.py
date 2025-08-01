import pandas as pd 
import torch 
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn 
from gensim.models import Word2Vec
import re
from collections import defaultdict

class DatasetcustomWord2Vec(Dataset):
    def __init__(self, csv_path, word2vec_model=None, fit_model=False, is_test=False):
        self.data = pd.read_csv(csv_path)
        self.is_test = is_test
    
        # Обработка бесконечных значений в salary_mean_net
        if 'salary_mean_net' in self.data.columns:
            # Заменяем inf на NaN, затем заполняем медианным значением
            self.data['salary_mean_net'] = self.data['salary_mean_net'].replace([np.inf, -np.inf], np.nan)
            median_salary = self.data['salary_mean_net'].median()
            self.data['salary_mean_net'] = self.data['salary_mean_net'].fillna(median_salary)
            
            # Нормализация целевой переменной для лучшей сходимости
            if not is_test:
                self.salary_mean = self.data['salary_mean_net'].mean()
                self.salary_std = self.data['salary_mean_net'].std()
                print(f"Нормализация salary: mean={self.salary_mean:.2f}, std={self.salary_std:.2f}")
                self.data['salary_mean_net'] = (self.data['salary_mean_net'] - self.salary_mean) / self.salary_std
            else:
                # Для тестового датасета используем значения из train
                self.salary_mean = 0  # Будет установлено из train_dataset
                self.salary_std = 1
    
        self.text_columns = [
            'name', 'employer_name', 'experience_name', 'schedule_name', 'key_skills_name',
            'unified_address_city', 'unified_address_state', 'unified_address_region', 'unified_address_country',
            'specializations_profarea_name', 'professional_roles_name', 'languages_name',
            'raw_description', 'lemmaized_wo_stopwords_raw_description', 'if_foreign_language',
            'is_branded_description', 'name_clean', 'employment_name', 'employer_industries'
        ]
        self.num_cols = ['accept_handicapped', 'accept_kids', 'employer_id']
        
        # Word2Vec параметры
        self.word2vec_dim = 64  # Размерность векторов слов
        self.max_words_per_col = 10  # Максимальное количество слов для каждого столбца
        
        if word2vec_model is None:
            self.word2vec_model = None
        else:
            self.word2vec_model = word2vec_model
            
        self._preprocess_data(fit_model)

    def _tokenize_text(self, text):
        """Токенизация текста"""
        if pd.isna(text):
            return []
        # Простая токенизация - можно улучшить
        text = str(text).lower()
        # Удаляем специальные символы, оставляем только буквы и цифры
        text = re.sub(r'[^а-яёa-z0-9\s]', ' ', text)
        words = text.split()
        return words[:self.max_words_per_col]  # Ограничиваем количество слов

    def _get_text_embedding(self, text, col_name):
        """Получение эмбеддинга для текста"""
        words = self._tokenize_text(text)
        if not words or self.word2vec_model is None:
            return np.zeros(self.word2vec_dim)
        
        word_vectors = []
        for word in words:
            try:
                word_vectors.append(self.word2vec_model.wv[word])
            except KeyError:
                continue
        
        if not word_vectors:
            return np.zeros(self.word2vec_dim)
        
        # Усредняем векторы всех слов
        return np.mean(word_vectors, axis=0)

    def _preprocess_data(self, fit_model):
        print("Подготовка текстовых данных для Word2Vec...")
        
        # Собираем все тексты для обучения Word2Vec
        all_texts = []
        for col in self.text_columns:
            texts = self.data[col].astype(str).fillna('')
            for text in texts:
                words = self._tokenize_text(text)
                if words:
                    all_texts.append(words)
        
        print(f"Всего текстов для Word2Vec: {len(all_texts)}")
        
        if fit_model and all_texts:
            print("Обучение Word2Vec модели...")
            self.word2vec_model = Word2Vec(
                sentences=all_texts,
                vector_size=self.word2vec_dim,
                window=5,
                min_count=1,
                workers=4,
                sg=1  # Skip-gram
            )
            print(f"Word2Vec модель обучена. Размер словаря: {len(self.word2vec_model.wv)}")
        
        # Создаем эмбеддинги для каждого столбца
        self.text_embeddings = []
        for col in self.text_columns:
            print(f"Создание эмбеддингов для {col}...")
            embeddings = []
            for text in self.data[col]:
                embedding = self._get_text_embedding(text, col)
                embeddings.append(embedding)
            self.text_embeddings.append(np.array(embeddings))
        
        self.text_embeddings = np.concatenate(self.text_embeddings, axis=1)
        print(f"Общее количество Word2Vec признаков: {self.text_embeddings.shape[1]}")
        
        # Обработка числовых признаков
        num_data = self.data[self.num_cols].copy()
        # Заменяем inf на NaN, затем заполняем медианными значениями
        for col in self.num_cols:
            num_data[col] = num_data[col].replace([np.inf, -np.inf], np.nan)
            median_val = num_data[col].median()
            num_data[col] = num_data[col].fillna(median_val)
        
        self.num_features = num_data.values.astype(np.float32)
        print(f"Количество числовых признаков: {self.num_features.shape[1]}")
        print(f"Общая размерность признаков: {self.text_embeddings.shape[1] + self.num_features.shape[1]}")
        
        # Проверка на NaN в признаках
        if np.any(np.isnan(self.text_embeddings)):
            print("ВНИМАНИЕ: Найдены NaN в Word2Vec признаках!")
            self.text_embeddings = np.nan_to_num(self.text_embeddings, nan=0.0)
        
        if np.any(np.isnan(self.num_features)):
            print("ВНИМАНИЕ: Найдены NaN в числовых признаках!")
            self.num_features = np.nan_to_num(self.num_features, nan=0.0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text_feat = self.text_embeddings[idx]
        num_feat = self.num_features[idx]
        features = np.concatenate([text_feat, num_feat]).astype(np.float32)
        
        # Дополнительная проверка на NaN в признаках
        if np.any(np.isnan(features)):
            features = np.nan_to_num(features, nan=0.0)
        
        if self.is_test:
            return self.data.iloc[idx]['id'], features
        else:
            target = self.data.iloc[idx]['salary_mean_net']
            # Проверка на NaN в целевой переменной
            if np.isnan(target):
                target = 0.0
            return features, target

class ModelforMoneyWord2Vec(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(16, 1)
        )
        
        # Инициализация весов для лучшей сходимости
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        return self.model(x)

def evaluate_model(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    valid_batches = 0
    with torch.no_grad():
        for X, y in data_loader:
            # Проверка на NaN в данных
            if torch.isnan(X).any() or torch.isnan(y).any():
                continue
            outputs = model(X)
            loss = criterion(outputs.squeeze(), y.float())
            if not torch.isnan(loss):
                total_loss += loss.item()
                valid_batches += 1
    return total_loss / valid_batches if valid_batches > 0 else float('inf')

def train(model, train_loader, test_loader, epochs=20, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.7)
    best_loss = float('inf')
    patience = 5  # Ранняя остановка
    patience_counter = 0
    
    print(f"Начинаем обучение с {epochs} эпохами, lr={lr}")
    print(f"Размер модели: {sum(p.numel() for p in model.parameters())} параметров")
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        valid_batches = 0
        
        for batch_idx, (X, y) in enumerate(train_loader):
            # Проверка на NaN в данных
            if torch.isnan(X).any() or torch.isnan(y).any():
                print(f"Пропускаем батч {batch_idx} из-за NaN")
                continue
                
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs.squeeze(), y.float())
            
            # Проверка на NaN в loss
            if torch.isnan(loss):
                print(f"NaN loss в батче {batch_idx}")
                continue
                
            loss.backward()
            
            # Gradient clipping для предотвращения взрыва градиентов
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_train_loss += loss.item()
            valid_batches += 1
            
            # Выводим прогресс каждые 100 батчей
            if batch_idx % 100 == 0 and batch_idx > 0:
                print(f"Эпоха {epoch+1}, батч {batch_idx}, loss: {loss.item():.2f}")
        
        if valid_batches == 0:
            print(f'Эпоха [{epoch+1}/{epochs}]: Нет валидных батчей')
            continue
            
        avg_train_loss = total_train_loss / valid_batches
        avg_test_loss = evaluate_model(model, test_loader, criterion)
        
        if avg_test_loss < best_loss and not np.isnan(avg_test_loss):
            best_loss = avg_test_loss
            torch.save(model.state_dict(), 'best_salary_model_word2vec.pth')
            print(f"Новая лучшая модель сохранена! Loss: {best_loss:.2f}")
            patience_counter = 0
        else:
            patience_counter += 1
        
        scheduler.step(avg_test_loss)
        
        # Выводим информацию каждую эпоху
        print(f'Эпоха [{epoch+1}/{epochs}]')
        print(f'Ошибка при обучении: {avg_train_loss:.2f}')
        print(f'Ошибка при тестировании: {avg_test_loss:.2f}')
        print(f'Лучшая ошибка: {best_loss:.2f}')
        print(f'Patience: {patience_counter}/{patience}')
        print('-' * 50)
        
        # Ранняя остановка
        if patience_counter >= patience:
            print(f"Ранняя остановка на эпохе {epoch+1}")
            break

def predict_and_save(model, test_dataset, train_dataset, output_path):
    test_tsv_path = '/Users/starfire/Desktop/зп вакансии/clean_test.csv'
    test_ids = pd.read_csv(test_tsv_path, usecols=['id'])['id'].astype(str).tolist()
    id_to_index = {str(test_dataset.data.iloc[i]['id']): i for i in range(len(test_dataset))}
    model.eval()
    results = []
    preds = []
    missing_indices = []
    found = 0
    
    # Получаем параметры нормализации из train_dataset
    salary_mean = train_dataset.salary_mean
    salary_std = train_dataset.salary_std
    print(f"Денормализация с параметрами: mean={salary_mean:.2f}, std={salary_std:.2f}")
    
    with torch.no_grad():
        for id_val in test_ids:
            idx = id_to_index.get(str(id_val), None)
            if idx is None:
                preds.append(np.nan)
                missing_indices.append(len(results))
                results.append({'id': id_val, 'salary_mean_net': np.nan})
            else:
                found += 1
                _, features = test_dataset[idx]
                if np.any(np.isnan(features)):
                    features = np.nan_to_num(features, nan=0.0)
                x = torch.FloatTensor(features).unsqueeze(0)
                pred = model(x).item()
                if np.isnan(pred):
                    pred = 0.0
                
                # Денормализация предсказания
                pred_denorm = pred * salary_std + salary_mean
                
                # Ограничиваем предсказания разумными пределами
                pred_denorm = max(0, pred_denorm)  # Не меньше 0
                pred_denorm = min(1000000, pred_denorm)  # Не больше 1M
                
                preds.append(pred_denorm)
                results.append({'id': id_val, 'salary_mean_net': pred_denorm})
    
    print(f"Всего id в test.tsv: {len(test_ids)}; найдено в test_dataset: {found}; отсутствует: {len(missing_indices)}")
    if missing_indices:
        print(f"Первые 5 отсутствующих id: {[test_ids[i] for i in missing_indices[:5]]}")
    
    # Заполняем пропуски медианой по предсказаниям
    median_pred = np.nanmedian([p for p in preds if not np.isnan(p)])
    for idx in missing_indices:
        results[idx]['salary_mean_net'] = median_pred
    
    df = pd.DataFrame(results)
    print(f"Пример первых 5 предсказаний: {df.head(5)}")
    print(f"Статистика предсказаний: min={df['salary_mean_net'].min():.2f}, max={df['salary_mean_net'].max():.2f}, mean={df['salary_mean_net'].mean():.2f}")
    df.to_csv(output_path, sep='\t', index=False)

if __name__ == '__main__':
    # Используем оригинальные датасеты
    train_path = '/Users/starfire/Desktop/зп вакансии/clean_train.csv'
    test_path = '/Users/starfire/Desktop/зп вакансии/clean_test.csv'
    sample_submission_path = '/Users/starfire/Desktop/зп вакансии/sample_submission_word2vec.tsv'
    
    print("Загрузка данных...")
    # Fit Word2Vec model on train, reuse for test
    train_dataset = DatasetcustomWord2Vec(train_path, fit_model=True)
    word2vec_model = train_dataset.word2vec_model
    test_dataset = DatasetcustomWord2Vec(test_path, word2vec_model=word2vec_model, is_test=True)
    
    input_dim = train_dataset[0][0].shape[0]
    print(f"Размерность входных признаков: {input_dim}")
    
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    
    model = ModelforMoneyWord2Vec(input_dim)
    print("Начинаем обучение модели...")
    train(model, train_loader, test_loader, epochs=20)
    
    print("Делаем предсказания...")
    # Диагностика перед predict_and_save
    print(f"train_dataset: {len(train_dataset)} samples, test_dataset: {len(test_dataset)} samples")
    train_ids = set(str(train_dataset.data['id'])) if 'id' in train_dataset.data.columns else set()
    test_ids = set(str(test_dataset.data['id'])) if 'id' in test_dataset.data.columns else set()
    print(f"train_dataset id count: {len(train_ids)}; test_dataset id count: {len(test_ids)}")
    
    predict_and_save(model, test_dataset, train_dataset, sample_submission_path)
    print("Готово!") 