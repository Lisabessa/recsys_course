"""
Семинар 3. Контентная фильтрация
Цель: Разработать методы контентной фильтрации по пользователям и по фильмам.
В качестве контента используем описание жанров для каждого фильма из movies.csv.
Для векторизации жанров используем CountVectorizer с разделителем "|".
"""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from utils import build_user_item_matrix, id_to_movie, load_data, print_user_rated_items


class ContentRecommender:
    """
    Класс для построения рекомендаций на основе контента - описания жанров.
    Матрица эмбеддингов размером (max_movie_id+1, n_genres), где строки
    соответствуют movieId, а столбцы — one-hot кодированию жанров.
    Матрица строится при инициализации экземпляра класса.
    """

    def __init__(self):
        self.embeddings = None
        self.ui_matrix = build_user_item_matrix()
        self._build_embeddings()

    def _build_embeddings(self):
        _, movies_df = load_data()
        self.movies_df = movies_df.copy()
        self.movies_df["genres"] = self.movies_df["genres"].fillna("")
        vectorizer = CountVectorizer(tokenizer=lambda s: s.split("|"), lowercase=False)

        genre_matrix = vectorizer.fit_transform(self.movies_df["genres"]).toarray()
        max_movie_id = int(self.movies_df["movieId"].max())
        embeddings = np.zeros((max_movie_id + 1, genre_matrix.shape[1]), dtype=float)

        for idx, row in self.movies_df.iterrows():
            mid = int(row["movieId"])
            embeddings[mid] = genre_matrix[idx]                      
        
        self.embeddings = embeddings
        

    def predict_rating(self, user_id: int, item_id: int, k: int = 5) -> float:
        """
        Предсказывает рейтинг user_id для item_id на основе контентной фильтрации.

        Алгоритм:
        1) Берём вектор целевого фильма: target_vec.
        2) Находим все фильмы, оцененные пользователем.
        3) Считаем косинусное сходство target_vec с векторами оцененных фильмов.
        4) Отбираем топ-k похожих оцененных фильмов (k-параметр).
        5) Предсказываем рейтинг как взвешенное среднее оценок по сходствам.
        6) Если не удаётся предсказать (нет оценок или нулевые векторы), возвращаем 0.0.
        7) Клипируем результат в [0.0, 5.0].

        Args:
            user_id: индекс пользователя
            item_id: индекс фильма
            k: сколько наиболее похожих оцененных фильмов использовать

        Returns:
            float: предсказанный рейтинг
        """
        target_vec = self.embeddings[item_id]

        user_ratings = self.ui_matrix[user_id]
        rated_indicies = np.where(user_ratings > 0)[0]
        if rated_indicies.size == 0:
            return 0.0
        
        rated_vecs = self.embeddings[rated_indicies]
        norms = np.linalg.norm(rated_vecs, axis=1)
        target_norm = np.linalg.norm(target_vec)

        valid = norms > 0
        if not np.any(valid):
            return 0.0
        
        rated_indicies = rated_indicies[valid]
        rated_vecs = rated_vecs[valid]
        norms = norms[valid]
        similarities = (rated_vecs @ target_vec) / (norms * target_norm + 1e-9)

        topk = min(k, similarities.shape[0])
        idx_sorted = np.argsort(similarities)[::-1][:topk]
        sim_topk = similarities[idx_sorted]
        rating_topk = user_ratings[rated_indicies[idx_sorted]]

        denom = np.sum(np.abs(sim_topk))
        if denom == 0:
            return 0.0
        pred = np.dot(sim_topk, rating_topk) / denom
        pred = float(np.clip(pred, 0.0, 5.0))
        return pred


    def predict_items_for_user(
        self, user_id: int, k: int = 5, n_recommendations: int = 5
    ) -> list:
        """
        Рекомендует фильмы пользователю user_id на основе контента фильма.

        Алгоритм:
        1) Берем все фильмы, которые оценил пользователь.
        3) Строим профиль пользователя как взвешенное среднее жанров оцененных фильмов.
        4) Для всех фильмов, которые пользователь не оценил, считаем сходство с профилем.
        5) Сортируем по убыванию сходства и возвращаем top-n.
        """
        n_users, n_items = self.ui_matrix.shape
        if user_id < 0 or user_id >= n_users:
            raise IndexError("user_id out of bounds")
        elif k <= 0:
            raise ValueError("k must be positive")
        elif n_recommendations <= 0:
            raise ValueError("n_recommendations must be > 0")
        
        user_ratings = self.ui_matrix[user_id]
        rated_indicies = np.where(user_ratings > 0)[0]
        if rated_indicies.size == 0:
            return []
        
        rated_vecs = self.embeddings[rated_indicies]
        weights = user_ratings[rated_indicies]
        if np.sum(weights) == 0:
            return []
        
        user_profile = np.average(rated_vecs, axis=0, weights=weights)
        profile_norm = np.linalg.norm(user_profile)
        if profile_norm == 0:
            return []
        
        unseen_indicies = np.where(user_ratings == 0)[0]
        if unseen_indicies.size == 0:
            return []
        
        unseen_vecs = self.embeddings[unseen_indicies]
        unseen_norms = np.linalg.norm(unseen_vecs, axis=1)
        valid = unseen_norms > 0
        if not np.any(valid):
            return []
        
        valid_indicies = unseen_indicies[valid]
        valid_vecs = unseen_vecs[valid]
        valid_norms = unseen_norms[valid]

        sims = (valid_vecs @ user_profile) / (valid_norms * profile_norm + 1e-9)
        top = np.argsort(sims)[::-1][:n_recommendations]
        return valid_indicies[top].tolist()


# Пример использования для дебага:
if __name__ == "__main__":
    user_id = 10
    item_id = 2
    k = 5
    content_recommender = ContentRecommender()
    print_user_rated_items(user_id, content_recommender.ui_matrix)

    pred_rating = content_recommender.predict_rating(user_id, item_id, k)
    print(f"Predicted rating for user {user_id} and item {item_id}: {pred_rating:.2f}")

    recommendations = content_recommender.predict_items_for_user(
        user_id, k=5, n_recommendations=10
    )
    for rec in recommendations:
        print(f"Recommended movie ID: {rec}, Title: {id_to_movie(rec)}")
