"""
Семинар 1. Простые рекомендации
Цель: построить базовые рекомендательные методы и оценить качество
на примере мобильной системы MovieLens.

Задачи:
1) Реализовать случайные рекомендации.
2) Реализовать рекомендации популярных фильмов на основе средних рейтингов.
3) Оценить системы по точности попадания в исторические оценки пользователя.

Для каждого метода требуется реализовать функцию, возвращающую
набор рекомендаций и метрику accuracy.
"""

import numpy as np

from utils import load_data


def random_recommend(n_recommendations: int = 10, seed: int = 42) -> list[int]:
    """
    Рекомендует случайные фильмы из всех доступных.

    Алгоритм:
    1) Загружаем рейтинговый DataFrame.
    2) Берём уникальные ID фильмов.
    3) Случайно выбираем n фильмов без повторов.

    Args:
        n_recommendations: Количество рекомендаций.
        seed: Seed для воспроизводимости.

    Returns:
        Список ID фильмов.
    """
    ratings_df, _ = load_data()
    np.random.seed(seed)
    all_movie_ids = ratings_df["movieId"].unique()
    recommendations = np.random.choice(
        all_movie_ids, size=n_recommendations, replace=False
    )
    return recommendations.tolist()


def top_n_recommend(
    n_recommendations: int = 10, min_ratings: int = 10
) -> list[tuple[int, float, int, str]]:
    """
    Рекомендует самые популярные фильмы по средней оценке и количеству оценок.

    Алгоритм:
    1) Загружаем данные ratings и movies.
    2) Группируем по movieId и считаем средний рейтинг и число оценок.
    3) Фильтруем фильмы с rating_count >= min_ratings.
    4) Сортируем по avg_rating и rating_count по убыванию.
    5) Берём top-n и добавляем названия фильмов.

    Args:
        n_recommendations: Количество рекомендаций.
        min_ratings: Мин. количество людей, которые оценили фильм.

    Returns:
        Список кортежей (movieId, avg_rating, rating_count, title).
    """
    raitings_df, movies_df = load_data()
    stats = (
        raitings_df.groupby("movieId").agg({"rating": ["mean", "count"]}).reset_index()
    )
    stats.columns = ["movieId", "avg_rating", "rating_count"]
    stats = stats[stats["rating_count"] >= min_ratings]
    chosen = stats.sort_values(
        ["avg_rating", "rating_count"], ascending=[False, False]
    ).head(n_recommendations)
    res = chosen.merge(movies_df[["movieId", "title"]], on="movieId")
    return[
        (row["movieId"], row["avg_rating"], row["rating_count"], row["title"])
        for _, row in res.iterrows()
    ]


def evaluate_rec_systems(
    user_id: int = 1, n_recommendations: int = 10, random_state: int = 42
) -> dict:
    """
    Оценивает эффективность базовых рекомендательных систем.

    Алгоритм:
    1) Получаем рекомендации случайных фильмов (random_recommend).
    2) Получаем рекомендации популярных фильмов (top_n_recommend).
    3) Берём исторические фильмы, которые пользователь уже оценил.
    4) Считаем Accuracy как долю рекомендованных фильмов,
       попавших в те фильмы, которые пользователь посмотрел.

    Args:
        user_id: ID пользователя.
        n_recommendations: Количество рекомендаций.
        random_state: Seed для случайных рекомендаций.

    Returns:
        Словарь {'random_accuracy', 'popular_accuracy'}.
    """
    ratings_df, _ = load_data()
    random_ids = random_recommend(
        n_recommendations=n_recommendations, seed=random_state
    )
    popular_recs = top_n_recommend(n_recommendations=n_recommendations)
    popular_ids = [rec[0] for rec in popular_recs]
    user_rated_movies = ratings_df[ratings_df["userId"] == user_id]["movieId"].tolist()
    relevant_items = set(user_rated_movies)
    random_accuracy = 0.0
    popular_accuracy = 0.0
    if len(relevant_items) > 0:
        random_accuracy = (len(set(random_ids) & relevant_items)) / n_recommendations
        popular_accuracy = (len(set(popular_ids) & relevant_items)) / n_recommendations
    return {"random_accuracy": random_accuracy, "popular_accuracy": popular_accuracy}


if __name__ == "__main__":
    # 1. Случайные рекомендации
    print("\n1. СЛУЧАЙНЫЕ РЕКОМЕНДАЦИИ:")
    print("-" * 60)
    random_recs = random_recommend(n_recommendations=10)
    print(f"Рекомендованные ID фильмов: {random_recs}")

    # 2. Популярные фильмы
    print("\n2. ПОПУЛЯРНЫЕ ФИЛЬМЫ (рекомендации на основе популярности):")
    print("-" * 60)
    popular_recs = top_n_recommend(n_recommendations=10)
    print(
        f"{'Rank':<5} {'ID':<6} {'Ср рейтинг':<18} {'Кол-во оценок':<15} {'Название'}"
    )
    print("-" * 60)
    for i, (movie_id, avg_rating, rating_count, title) in enumerate(popular_recs, 1):
        print(
            f"{i:<5} {movie_id:<6} {avg_rating:<18.2f} {rating_count:<15} {title[:50]}"
        )

    # 3. Оценка системы
    print("\n3. ОЦЕНКА КАЧЕСТВА СИСТЕМЫ:")
    print("-" * 60)
    metrics = evaluate_rec_systems()
    print(f"Accuracy (случайные рекомендации): {metrics['random_accuracy']:.4f}")
    print(f"Accuracy (популярные фильмы): {metrics['popular_accuracy']:.4f}")
