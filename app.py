import os
import random
import re
import traceback
import unicodedata

import pandas as pd
import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
load_dotenv()

TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_BASE_URL = "https://api.themoviedb.org/3"
IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"

RUSSIAN_LETTERS = set("абвгдеёжзийклмнопрстуфхцчшщъыьэюя")

GENRE_TRANSLATIONS = {
    "Adventure": "Приключения",
    "Comedy": "Комедия",
    "Crime": "Криминал",
    "Documentary": "Документальный",
    "Drama": "Драма",
    "Family": "Семейный",
    "Fantasy": "Фэнтези",
    "Horror": "Ужасы",
    "Musical": "Музыкальный",
    "Mystery": "Мистика",
    "Romance": "Романтика",
}

TMDB_GENRE_IDS = {
    "Adventure": 12,
    "Comedy": 35,
    "Crime": 80,
    "Documentary": 99,
    "Drama": 18,
    "Family": 10751,
    "Fantasy": 14,
    "Horror": 27,
    "Musical": 10402,
    "Mystery": 9648,
    "Romance": 10749,
}

movie_data = None
genres = []
age_ratings = []


def infer_age_rating(genres_text="", description_text="") -> str:
    """Грубая эвристика возрастного рейтинга по жанрам и описанию."""
    text = f"{genres_text or ''} {description_text or ''}".lower()

    if any(word in text for word in [
        'horror', 'ужасы', 'slasher', 'gore', 'blood', 'bloody', 'violent',
        'serial killer', 'murder', 'rape', 'torture', 'drug cartel'
    ]):
        return '18+'

    if any(word in text for word in [
        'crime', 'криминал', 'thriller', 'триллер', 'mystery', 'мистика',
        'detective', 'drama', 'драма', 'war', 'война'
    ]):
        return '16+'

    if any(word in text for word in [
        'adventure', 'приключения', 'fantasy', 'фэнтези', 'romance', 'романтика',
        'comedy', 'комедия', 'action', 'боевик'
    ]):
        return '12+'

    if any(word in text for word in [
        'family', 'семейный', 'musical', 'музыкальный', 'documentary', 'документальный',
        'animation', 'animated', 'мульт'
    ]):
        return '6+'

    return '0+'


def map_tmdb_certification_to_age(certification: str):
    cert = (certification or "").upper().strip()
    mapping = {
        "G": "0+",
        "TV-G": "0+",
        "TV-Y": "0+",
        "TV-Y7": "6+",
        "PG": "6+",
        "TV-PG": "12+",
        "PG-13": "12+",
        "TV-14": "16+",
        "R": "18+",
        "NC-17": "18+",
        "TV-MA": "18+",
    }
    return mapping.get(cert)


def remove_duplicate_movies(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    result = df.copy()

    if 'name_normalized' not in result.columns and 'Name' in result.columns:
        result['name_normalized'] = result['Name'].apply(normalize_text)

    dedupe_cols = []
    if 'name_normalized' in result.columns:
        dedupe_cols.append('name_normalized')
    elif 'Name' in result.columns:
        dedupe_cols.append('Name')

    if 'year' in result.columns:
        dedupe_cols.append('year')

    if dedupe_cols:
        result = result.drop_duplicates(subset=dedupe_cols, keep='first')
    else:
        result = result.drop_duplicates()

    return result


def choose_diverse_movies(df: pd.DataFrame, limit: int = 12, min_rating: float = 4.3) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    result = df.copy()
    result = remove_duplicate_movies(result)

    if 'RatingValue' in result.columns:
        result['RatingValue'] = pd.to_numeric(result['RatingValue'], errors='coerce')
    else:
        result['RatingValue'] = pd.NA

    if 'popularity' in result.columns:
        result['popularity'] = pd.to_numeric(result['popularity'], errors='coerce').fillna(0)
    else:
        result['popularity'] = 0

    result = result[result['RatingValue'].fillna(0) >= min_rating]
    if result.empty:
        return result

    result = result.sort_values(by=['RatingValue', 'popularity'], ascending=False, na_position='last')

    pool_size = min(max(limit * 5, 30), len(result))
    pool = result.head(pool_size).copy()

    if len(pool) <= limit:
        return pool

    sampled = pool.sample(n=limit, replace=False, random_state=random.randint(1, 10**9))
    sampled = sampled.sort_values(by=['RatingValue', 'popularity'], ascending=False, na_position='last')
    return sampled


def contains_russian(text: str) -> bool:
    if not text or not isinstance(text, str):
        return False
    return any(ch.lower() in RUSSIAN_LETTERS for ch in text)


def normalize_text(text) -> str:
    if pd.isna(text) or text is None or not isinstance(text, str) or not text.strip():
        return ""
    text = unicodedata.normalize("NFKD", str(text)).encode("ASCII", "ignore").decode("ASCII")
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def check_files_exist():
    files = ["final_data.csv", "wiki_movie_plots_deduped.csv"]
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        raise FileNotFoundError(
            f"Отсутствуют необходимые файлы: {', '.join(missing)}. "
            "Поместите final_data.csv и wiki_movie_plots_deduped.csv в папку проекта."
        )


def load_and_prepare_data() -> pd.DataFrame:
    check_files_exist()

    df1 = pd.read_csv("final_data.csv")
    df2 = pd.read_csv("wiki_movie_plots_deduped.csv")

    if "DatePublished" in df1.columns:
        df1["year"] = df1["DatePublished"].apply(
            lambda x: str(x)[:4] if pd.notnull(x) and str(x).strip() else "0"
        )
        df1 = df1[pd.to_numeric(df1["year"], errors="coerce") >= 1980]

    if "Release Year" in df2.columns:
        df2 = df2[pd.to_numeric(df2["Release Year"], errors="coerce") >= 1980]

    if "Name" not in df1.columns:
        raise ValueError("В final_data.csv отсутствует колонка 'Name'.")

    df1["name_normalized"] = df1["Name"].apply(normalize_text)

    if "Title" in df2.columns:
        df2["title_normalized"] = df2["Title"].apply(normalize_text)
    elif "Film" in df2.columns:
        df2["title_normalized"] = df2["Film"].apply(normalize_text)
    else:
        raise ValueError("Во втором CSV должна быть колонка 'Title' или 'Film'.")

    merged_df = pd.merge(df1, df2, left_on="name_normalized", right_on="title_normalized", how="inner")
    if merged_df.empty:
        raise ValueError("Не удалось объединить CSV-файлы по названиям фильмов.")

    merged_df["keywords_text"] = merged_df.apply(
        lambda row: " ".join(
            filter(
                None,
                [
                    normalize_text(row["Keywords"]) if "Keywords" in row and pd.notnull(row["Keywords"]) else "",
                    normalize_text(row["Genres"]) if "Genres" in row and pd.notnull(row["Genres"]) else "",
                    normalize_text(row["Actors"]) if "Actors" in row and pd.notnull(row["Actors"]) else "",
                    normalize_text(row["Director"]) if "Director" in row and pd.notnull(row["Director"]) else "",
                    normalize_text(row["Name"]) if "Name" in row and pd.notnull(row["Name"]) else "",
                ],
            )
        ),
        axis=1,
    )

    merged_df["plot_text"] = merged_df.apply(
        lambda row: normalize_text(row["Plot"]) if "Plot" in row and pd.notnull(row["Plot"]) else "",
        axis=1,
    )
    merged_df["combined_text"] = merged_df["keywords_text"] + " " + merged_df["plot_text"]

    if "RatingValue" in merged_df.columns:
        merged_df["RatingValue"] = pd.to_numeric(merged_df["RatingValue"], errors="coerce")
    else:
        merged_df["RatingValue"] = pd.NA

    if "popularity" in merged_df.columns:
        merged_df["popularity"] = pd.to_numeric(merged_df["popularity"], errors="coerce").fillna(0)
    else:
        merged_df["popularity"] = 0

    merged_df["AgeRating"] = merged_df.apply(
        lambda row: infer_age_rating(row.get("Genres", ""), row.get("Description", "")),
        axis=1,
    )

    merged_df = remove_duplicate_movies(merged_df)

    merged_df.attrs["genres"] = list(GENRE_TRANSLATIONS.keys())
    merged_df.attrs["age_ratings"] = ["0+", "6+", "12+", "16+", "18+"]
    return merged_df


def _age_columns(df: pd.DataFrame):
    possible = [
        "AgeRating",
        "age_rating",
        "Age Rating",
        "ContentRating",
        "contentRating",
        "Certificate",
        "certificate",
    ]
    return [col for col in possible if col in df.columns]


def apply_local_filters(df: pd.DataFrame, genre=None, age_rating=None) -> pd.DataFrame:
    filtered = df.copy()

    if genre and "Genres" in filtered.columns:
        filtered = filtered[
            filtered["Genres"].fillna("").str.contains(re.escape(str(genre)), case=False, na=False)
        ]

    if age_rating:
        age_cols = _age_columns(filtered)
        if age_cols:
            normalized_age = str(age_rating).strip()
            mask = False
            for col in age_cols:
                current = filtered[col].fillna("").astype(str).str.strip().eq(normalized_age)
                mask = current if isinstance(mask, bool) else (mask | current)
            filtered = filtered[mask]

    return filtered


def format_local_results(df: pd.DataFrame, limit: int = 12, min_rating: float = 4.3):
    if df is None or df.empty:
        return []

    df = choose_diverse_movies(df, limit=limit, min_rating=min_rating)
    if df is None or df.empty:
        return []
    results = []
    for _, movie in df.iterrows():
        rating = None
        if pd.notnull(movie.get("RatingValue")):
            try:
                rating = round(float(movie.get("RatingValue")), 1)
            except Exception:
                rating = None

        description = (
            movie.get("Description")
            if pd.notnull(movie.get("Description", ""))
            else movie.get("Plot", "")
        )
        if pd.isna(description):
            description = ""

        actors = movie.get("Actors", "")
        if pd.isna(actors):
            actors = ""

        results.append(
            {
                "id": movie.get("id"),
                "title": movie.get("Name", "Без названия"),
                "poster": movie.get("PosterLink", "") if pd.notnull(movie.get("PosterLink", "")) else "",
                "rating": rating,
                "popularity": float(movie.get("popularity", 0) or 0),
                "genres": movie.get("Genres", ""),
                "description": str(description).strip(),
                "age_rating": movie.get("AgeRating", ""),
                "actors": str(actors).strip(),
            }
        )
    return results


def filter_movies(data: pd.DataFrame, genre=None, age_rating=None, limit: int = 12):
    filtered = apply_local_filters(data, genre, age_rating)
    filtered = remove_duplicate_movies(filtered)
    return format_local_results(filtered, limit=limit, min_rating=4.3)


def _search_by_field(query: str, data: pd.DataFrame, field: str, genre=None, age_rating=None, label="search"):
    if data.empty or field not in data.columns:
        return []

    normalized_query = normalize_text(query)
    if not normalized_query:
        return []

    vectorizer = TfidfVectorizer(stop_words="english", min_df=2, max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(data[field])
    query_vec = vectorizer.transform([normalized_query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[::-1][:20]

    filtered_rows = []
    for idx in top_indices:
        if similarities[idx] <= 0.1:
            continue
        row = data.iloc[idx:idx + 1]
        row = apply_local_filters(row, genre, age_rating)
        if row.empty:
            continue
        filtered_rows.append(row.iloc[0])

    if not filtered_rows:
        return []

    result_df = pd.DataFrame(filtered_rows)
    result_df = remove_duplicate_movies(result_df)
    results = format_local_results(result_df, limit=12, min_rating=4.3)
    for item in results:
        item["search_type"] = label
    return results


def keyword_search(query, data, genre=None, age_rating=None):
    return _search_by_field(query, data, "keywords_text", genre, age_rating, "keyword")


def plot_search(query, data, genre=None, age_rating=None):
    return _search_by_field(query, data, "plot_text", genre, age_rating, "plot")


def combined_search(query, data, genre=None, age_rating=None):
    return _search_by_field(query, data, "combined_text", genre, age_rating, "combined")


def dedupe_results(results, limit=12):
    seen = set()
    unique = []
    for item in results:
        raw_title = item.get("title") or ""
        title_key = normalize_text(raw_title)
        key = item.get("id") or title_key or raw_title
        if key in seen:
            continue
        if (item.get("rating") or 0) < 4.3:
            continue
        seen.add(key)
        unique.append(item)

    if len(unique) > limit:
        unique.sort(key=lambda x: ((x.get("rating") or 0), (x.get("popularity") or 0)), reverse=True)
        pool_size = min(max(limit * 5, 30), len(unique))
        pool = unique[:pool_size]
        unique = random.sample(pool, limit)

    unique.sort(key=lambda x: ((x.get("rating") or 0), (x.get("popularity") or 0)), reverse=True)
    return unique[:limit]


def get_movie_details(movie_id):
    if not TMDB_API_KEY:
        return {}
    try:
        response = requests.get(
            f"{TMDB_BASE_URL}/movie/{movie_id}",
            params={
                "api_key": TMDB_API_KEY,
                "language": "ru-RU",
                "append_to_response": "release_dates,credits",
            },
            timeout=15,
        )
        response.raise_for_status()
        details = response.json()

        age_rating = None
        for country_block in details.get("release_dates", {}).get("results", []):
            if country_block.get("iso_3166_1") in {"US", "RU"}:
                for item in country_block.get("release_dates", []):
                    cert = item.get("certification")
                    age_rating = map_tmdb_certification_to_age(cert)
                    if age_rating:
                        break
            if age_rating:
                break

        if not age_rating:
            age_rating = infer_age_rating(
                ", ".join(g.get("name", "") for g in details.get("genres", [])),
                details.get("overview", ""),
            )

        details["age_rating"] = age_rating
        return details
    except Exception:
        return {}


def tmdb_search(query=None, genre=None, age_rating=None):
    if not TMDB_API_KEY:
        return []

    params = {
        "api_key": TMDB_API_KEY,
        "language": "ru-RU",
        "include_adult": "false",
        "page": 1,
    }

    genre_id = TMDB_GENRE_IDS.get(genre) if genre else None
    if genre_id:
        params["with_genres"] = genre_id
    if age_rating:
        params["certification_country"] = "US"
        params["certification"] = age_rating

    try:
        if query:
            params["query"] = query
            response = requests.get(f"{TMDB_BASE_URL}/search/movie", params=params, timeout=15)
        else:
            params["sort_by"] = "popularity.desc"
            response = requests.get(f"{TMDB_BASE_URL}/discover/movie", params=params, timeout=15)

        response.raise_for_status()
        data = response.json()

        movies = []
        for movie in data.get("results", []):
            release_date = movie.get("release_date") or ""
            if release_date:
                try:
                    if int(release_date[:4]) < 1980:
                        continue
                except Exception:
                    pass

            details = get_movie_details(movie["id"])
            cast_names = []
            for person in details.get("credits", {}).get("cast", [])[:5]:
                name = person.get("name")
                if name:
                    cast_names.append(name)

            movies.append(
                {
                    "id": movie.get("id"),
                    "title": movie.get("title", movie.get("name", "N/A")),
                    "poster": f"{IMAGE_BASE_URL}{movie['poster_path']}" if movie.get("poster_path") else "",
                    "rating": movie.get("vote_average", 0),
                    "popularity": movie.get("popularity", 0),
                    "genres": ", ".join(g["name"] for g in details.get("genres", [])) if details else "",
                    "description": movie.get("overview", "") or (details.get("overview", "") if details else ""),
                    "age_rating": details.get("age_rating", "") if details else "",
                    "actors": ", ".join(cast_names),
                }
            )

        movies.sort(key=lambda x: (x.get("popularity", 0), x.get("rating", 0)), reverse=True)
        return movies[:12]
    except Exception as exc:
        print(f"Ошибка при поиске через TMDB: {exc}")
        return []


def initialize_app():
    global movie_data, genres, age_ratings
    try:
        movie_data = load_and_prepare_data()
        genres = movie_data.attrs.get("genres", list(GENRE_TRANSLATIONS.keys()))
        age_ratings = movie_data.attrs.get("age_ratings", ["0+", "6+", "12+", "16+", "18+"])
        print(f"Инициализация завершена. Загружено {len(movie_data)} фильмов.")
        return True
    except Exception as exc:
        print(f"КРИТИЧЕСКАЯ ОШИБКА: {exc}")
        movie_data = None
        genres = list(GENRE_TRANSLATIONS.keys())
        age_ratings = ["0+", "6+", "12+", "16+", "18+"]
        return False


@app.route("/")
def index():
    return render_template(
        "index.html",
        genres=genres,
        age_ratings=age_ratings,
        genre_translations=GENRE_TRANSLATIONS,
    )


@app.route("/health")
def health():
    return jsonify({"success": True, "status": "ok"})


@app.route("/search", methods=["POST"])
def search():
    try:
        data = request.get_json(silent=True) or {}
        query = (data.get("query") or "").strip()
        genre = data.get("genre") or None
        age_rating = data.get("age_rating") or None

        print(f"Получен запрос: '{query}'")
        print(f"Фильтры: жанр={genre}, возраст={age_rating}")

        if not query and not genre and not age_rating:
            if movie_data is not None:
                results = filter_movies(movie_data, None, None, limit=12)
                return jsonify({
                    "success": True,
                    "query": query,
                    "search_type": "random_high_rating",
                    "results": results,
                })

            if TMDB_API_KEY:
                results = tmdb_search(None, None, None)
                return jsonify({
                    "success": True,
                    "query": query,
                    "search_type": "tmdb_random_high_rating",
                    "results": results,
                })

            return jsonify({"error": "Нет локальных данных и не настроен TMDB API ключ."}), 500

        if not query and (genre or age_rating):
            if movie_data is not None:
                results = filter_movies(movie_data, genre, age_rating, limit=12)
                return jsonify({
                    "success": True,
                    "query": query,
                    "search_type": "filter",
                    "results": results,
                })

            if TMDB_API_KEY:
                results = tmdb_search(None, genre, age_rating)
                return jsonify({
                    "success": True,
                    "query": query,
                    "search_type": "tmdb_filter",
                    "results": results,
                })

            return jsonify({"error": "Нет локальных данных и не настроен TMDB API ключ."}), 500

        if contains_russian(query):
            results = tmdb_search(query, genre, age_rating)
            return jsonify({
                "success": True,
                "query": query,
                "search_type": "tmdb",
                "results": results,
            })

        if movie_data is None:
            return jsonify({"error": "Локальные данные не загружены. Проверьте CSV-файлы."}), 500

        results = combined_search(query, movie_data, genre, age_rating)
        if len(results) < 5:
            results.extend(keyword_search(query, movie_data, genre, age_rating))
            results.extend(plot_search(query, movie_data, genre, age_rating))

        results = dedupe_results(results, limit=12)
        return jsonify({
            "success": True,
            "query": query,
            "search_type": "local",
            "results": results,
        })

    except Exception as exc:
        error_details = traceback.format_exc()
        print(f"Ошибка при поиске: {exc}")
        print(error_details)
        return jsonify({
            "error": f"Произошла ошибка при поиске: {exc}",
            "details": error_details if app.debug else None,
        }), 500


@app.errorhandler(404)
def page_not_found(e):
    return render_template("error.html", error_code=404, error_message="Страница не найдена"), 404


@app.errorhandler(500)
def internal_server_error(e):
    return render_template("error.html", error_code=500, error_message="Внутренняя ошибка сервера"), 500


INDEX_HTML = '''<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ФильмоПоиск - Найдите идеальный фильм</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        :root {
            --primary-bg: #0f172a;
            --card-bg: #1e293b;
            --text-color: #f1f5f9;
            --accent-color: #3b82f6;
            --accent-dark: #1d4ed8;
            --error-color: #f43f5e;
            --filter-bg: #334155;
        }

        body {
            background-color: var(--primary-bg);
            color: var(--text-color);
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            min-height: 100vh;
        }

        .header {
            text-align: center;
            padding: 2rem 0;
            margin-bottom: 2rem;
        }

        .logo {
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(45deg, #ff6b6b, #4dabf7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .search-container {
            max-width: 860px;
            margin: 0 auto 3rem;
            background: rgba(30, 41, 59, 0.9);
            padding: 2rem;
            border-radius: 18px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.45);
            border: 1px solid rgba(100, 150, 255, 0.25);
        }

        .search-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 1rem;
            margin-bottom: 1rem;
        }

        .active-filters {
            font-size: 0.95rem;
            color: #cbd5e1;
        }

        .filter-btn {
            background: var(--filter-bg);
            color: white;
            border: none;
            padding: 0.6rem 1rem;
            border-radius: 10px;
            cursor: pointer;
        }

        .search-box {
            position: relative;
            display: flex;
            align-items: center;
        }

        #searchInput {
            width: 100%;
            padding: 1rem 1rem 1rem 2.8rem;
            border: 1px solid rgba(100, 150, 255, 0.45);
            border-radius: 50px;
            background: rgba(35, 40, 60, 0.95);
            color: white;
            font-size: 1.05rem;
        }

        #searchInput:focus {
            outline: none;
            box-shadow: 0 0 0 0.2rem rgba(77, 171, 247, 0.25);
        }

        .search-icon {
            position: absolute;
            left: 1rem;
            color: #60a5fa;
        }

        .search-btn {
            width: 100%;
            background: linear-gradient(120deg, #4dabf7, #339af0);
            color: white;
            border: none;
            padding: 0.95rem;
            border-radius: 50px;
            font-size: 1.05rem;
            font-weight: 600;
            margin-top: 1rem;
            cursor: pointer;
            box-shadow: 0 4px 15px rgba(77, 171, 247, 0.4);
        }

        .filter-modal {
            position: fixed;
            inset: 0;
            background: rgba(0,0,0,0.7);
            display: none;
            justify-content: center;
            align-items: center;
            z-index: 1000;
        }

        .filter-content {
            background: var(--card-bg);
            border-radius: 18px;
            width: 90%;
            max-width: 640px;
            padding: 2rem;
            position: relative;
        }

        .close-filter {
            position: absolute;
            top: 1rem;
            right: 1rem;
            background: none;
            border: none;
            color: white;
            font-size: 1.75rem;
            cursor: pointer;
        }

        .filter-title {
            margin-bottom: 1.5rem;
            color: var(--accent-color);
        }

        .filter-section {
            margin-bottom: 1.75rem;
        }

        .filter-options {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
            gap: 0.75rem;
        }

        .filter-option {
            background: var(--filter-bg);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 10px;
            padding: 0.8rem;
            text-align: center;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .filter-option:hover, .filter-option.selected {
            background: var(--accent-dark);
            border-color: var(--accent-color);
            transform: translateY(-2px);
        }

        .results-container {
            display: none;
            margin-top: 2rem;
        }

        .movie-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(230px, 1fr));
            gap: 1.25rem;
        }

        .movie-card {
            background: var(--card-bg);
            border-radius: 16px;
            overflow: hidden;
            border: 1px solid rgba(100, 150, 255, 0.2);
            box-shadow: 0 5px 15px rgba(0,0,0,0.28);
            cursor: pointer;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }

        .movie-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 10px 24px rgba(0,0,0,0.34);
        }

        .movie-poster-container {
            height: 340px;
            overflow: hidden;
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        }

        .movie-poster {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }

        .movie-poster-placeholder {
            height: 100%;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            color: #94a3b8;
        }

        .placeholder-icon {
            font-size: 3rem;
            margin-bottom: 0.75rem;
        }

        .movie-info {
            padding: 1rem;
        }

        .movie-title {
            font-size: 1.1rem;
            font-weight: 700;
            color: white;
            margin-bottom: 0.5rem;
            min-height: 2.6rem;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
            overflow: hidden;
        }

        .movie-meta {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 0.5rem;
            flex-wrap: wrap;
        }

        .movie-rating {
            display: inline-block;
            background: linear-gradient(120deg, #f6d365, #fda085);
            color: #1a1a2e;
            padding: 0.2rem 0.8rem;
            border-radius: 15px;
            font-weight: bold;
            font-size: 0.95rem;
        }

        .movie-genres {
            color: #cbd5e1;
            font-size: 0.85rem;
        }

        .movie-age {
            display: inline-block;
            background: rgba(59, 130, 246, 0.18);
            color: #bfdbfe;
            border: 1px solid rgba(59, 130, 246, 0.35);
            padding: 0.2rem 0.55rem;
            border-radius: 12px;
            font-size: 0.8rem;
            margin-top: 0.55rem;
        }

        .movie-description-modal {
            position: fixed;
            inset: 0;
            background: rgba(0, 0, 0, 0.72);
            display: none;
            justify-content: center;
            align-items: center;
            z-index: 1100;
            padding: 1rem;
        }

        .movie-description-content {
            width: 100%;
            max-width: 720px;
            background: var(--card-bg);
            border-radius: 18px;
            padding: 1.5rem;
            position: relative;
            border: 1px solid rgba(100, 150, 255, 0.22);
        }

        .movie-description-title {
            margin-bottom: 0.8rem;
        }

        .movie-description-layout {
            display: flex;
            gap: 1rem;
            flex-wrap: wrap;
            align-items: flex-start;
        }

        .movie-description-poster {
            width: 220px;
            max-width: 100%;
            border-radius: 14px;
            object-fit: cover;
            background: #0f172a;
            border: 1px solid rgba(100, 150, 255, 0.2);
        }

        .movie-description-main {
            flex: 1;
            min-width: 260px;
        }

        .movie-description-text {
            color: #e2e8f0;
            line-height: 1.6;
            white-space: pre-wrap;
        }

        .movie-description-meta {
            display: flex;
            gap: 0.75rem;
            flex-wrap: wrap;
            margin-bottom: 1rem;
        }

        .movie-description-actors {
            color: #cbd5e1;
            margin-bottom: 0.9rem;
            line-height: 1.5;
        }

        .loading, .error-container {
            display: none;
            text-align: center;
            padding: 3rem;
        }

        .spinner {
            width: 50px;
            height: 50px;
            border: 5px solid rgba(77,171,247,0.25);
            border-top: 5px solid #4dabf7;
            border-radius: 50%;
            margin: 0 auto 1rem;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }

        .error-container {
            background: rgba(220,53,69,0.15);
            border-radius: 15px;
            border: 1px solid rgba(220,53,69,0.3);
            max-width: 860px;
            margin: 2rem auto;
        }

        .footer {
            text-align: center;
            color: #64748b;
            padding: 2rem 0;
            margin-top: 2rem;
            border-top: 1px solid rgba(100,150,255,0.2);
        }
    </style>
</head>
<body>
    <div class="container py-4">
        <header class="header">
            <h1 class="logo">🎬 ФильмоПоиск</h1>
            <p class="lead text-center text-muted">Поиск фильмов по запросу, жанру и возрастной категории</p>
        </header>

        <main>
            <div class="search-container">
                <div class="search-header">
                    <button id="filterBtn" class="filter-btn">🔍 Фильтры</button>
                    <div id="activeFilters" class="active-filters">Фильтры не выбраны</div>
                </div>

                <div class="search-box">
                    <span class="search-icon">🔍</span>
                    <input type="text" id="searchInput" class="form-control" placeholder="Например: детектив, космос, Леонардо ДиКаприо...">
                </div>
                <button id="searchBtn" class="search-btn">🎬 Найти фильмы</button>
            </div>

            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p class="h5">Ищем фильмы...</p>
                <p class="text-muted" id="searchQueryDisplay"></p>
            </div>

            <div class="results-container" id="resultsContainer">
                <div class="results-header text-center mb-4">
                    <h2 class="h3">Результаты</h2>
                </div>
                <div class="movie-grid" id="moviesGrid"></div>
            </div>

            <div class="error-container" id="errorContainer">
                <h3 class="text-danger mb-3">⚠️ Ошибка</h3>
                <p id="errorMessage" class="mb-0"></p>
            </div>
        </main>
    </div>

    <div class="filter-modal" id="filterModal">
        <div class="filter-content">
            <button class="close-filter" id="closeFilter">&times;</button>
            <h3 class="filter-title">Фильтры</h3>

            <div class="filter-section">
                <h4>Жанры</h4>
                <div class="filter-options" id="genreOptions">
                    {% for genre in genres %}
                    <div class="filter-option" data-value="{{ genre }}">{{ genre_translations.get(genre, genre) }}</div>
                    {% endfor %}
                </div>
            </div>

            <div class="filter-section">
                <h4>Возрастные категории</h4>
                <div class="filter-options" id="ageOptions">
                    {% for age in age_ratings %}
                    <div class="filter-option" data-value="{{ age }}">{{ age }}</div>
                    {% endfor %}
                </div>
            </div>

            <div class="d-flex gap-2 mt-3">
                <button class="search-btn mt-0" id="applyFilters">Применить фильтры</button>
                <button class="search-btn mt-0" id="resetFilters" style="background: linear-gradient(120deg, #64748b, #475569); box-shadow: none;">Сбросить</button>
            </div>
        </div>
    </div>

    <footer class="footer">
        <p class="mb-0">© <span id="year"></span> ФильмоПоиск</p>
    </footer>

    <div class="movie-description-modal" id="movieDescriptionModal">
        <div class="movie-description-content">
            <button class="close-filter" id="closeMovieDescription">&times;</button>
            <div class="movie-description-layout">
                <img id="movieDescriptionPoster" class="movie-description-poster" alt="Постер фильма">
                <div class="movie-description-main">
                    <h3 class="movie-description-title" id="movieDescriptionTitle">Описание фильма</h3>
                    <div class="movie-description-meta" id="movieDescriptionMeta"></div>
                    <div class="movie-description-actors" id="movieDescriptionActors"></div>
                    <div class="movie-description-text" id="movieDescriptionText"></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        document.getElementById('year').textContent = new Date().getFullYear();

        const searchInput = document.getElementById('searchInput');
        const searchBtn = document.getElementById('searchBtn');
        const loading = document.getElementById('loading');
        const resultsContainer = document.getElementById('resultsContainer');
        const moviesGrid = document.getElementById('moviesGrid');
        const errorContainer = document.getElementById('errorContainer');
        const errorMessage = document.getElementById('errorMessage');
        const searchQueryDisplay = document.getElementById('searchQueryDisplay');
        const filterBtn = document.getElementById('filterBtn');
        const filterModal = document.getElementById('filterModal');
        const closeFilter = document.getElementById('closeFilter');
        const applyFiltersBtn = document.getElementById('applyFilters');
        const resetFiltersBtn = document.getElementById('resetFilters');
        const activeFilters = document.getElementById('activeFilters');
        const movieDescriptionModal = document.getElementById('movieDescriptionModal');
        const closeMovieDescriptionBtn = document.getElementById('closeMovieDescription');
        const movieDescriptionTitle = document.getElementById('movieDescriptionTitle');
        const movieDescriptionMeta = document.getElementById('movieDescriptionMeta');
        const movieDescriptionActors = document.getElementById('movieDescriptionActors');
        const movieDescriptionText = document.getElementById('movieDescriptionText');
        const movieDescriptionPoster = document.getElementById('movieDescriptionPoster');

        let currentFilters = { genre: null, age_rating: null };

        function updateActiveFiltersLabel() {
            const parts = [];
            if (currentFilters.genre) parts.push(`жанр: ${currentFilters.genre}`);
            if (currentFilters.age_rating) parts.push(`возраст: ${currentFilters.age_rating}`);
            activeFilters.textContent = parts.length ? parts.join(' · ') : 'Фильтры не выбраны';
        }

        function showLoading(text) {
            searchQueryDisplay.textContent = text || 'Выполняется поиск';
            loading.style.display = 'block';
            resultsContainer.style.display = 'none';
            errorContainer.style.display = 'none';
        }

        function hideLoading() {
            loading.style.display = 'none';
        }

        function showError(message) {
            errorMessage.textContent = message;
            errorContainer.style.display = 'block';
            resultsContainer.style.display = 'none';
            hideLoading();
        }

        function escapeHtml(value) {
            return String(value || '')
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;')
                .replace(/"/g, '&quot;')
                .replace(/'/g, '&#039;');
        }

        function openMovieDescription(movie) {
            movieDescriptionTitle.textContent = movie.title || 'Описание фильма';

            const metaParts = [];
            if (movie.rating) metaParts.push(`<span class="movie-rating">⭐ ${movie.rating}</span>`);
            if (movie.age_rating) metaParts.push(`<span class="movie-age">${escapeHtml(movie.age_rating)}</span>`);
            if (movie.genres) metaParts.push(`<span class="movie-genres">${escapeHtml(movie.genres)}</span>`);
            movieDescriptionMeta.innerHTML = metaParts.join('');

            movieDescriptionActors.innerHTML = (movie.actors && movie.actors.trim())
                ? `<strong>Актеры:</strong> ${escapeHtml(movie.actors)}`
                : '';

            movieDescriptionText.textContent = (movie.description && movie.description.trim())
                ? movie.description.trim()
                : 'Краткое описание пока отсутствует.';

            if (movie.poster && movie.poster.trim() !== '') {
                movieDescriptionPoster.src = movie.poster;
                movieDescriptionPoster.style.display = 'block';
            } else {
                movieDescriptionPoster.removeAttribute('src');
                movieDescriptionPoster.style.display = 'none';
            }

            movieDescriptionModal.style.display = 'flex';
        }

        function closeMovieDescription() {
            movieDescriptionModal.style.display = 'none';
        }

        function showResults(results) {
            moviesGrid.innerHTML = '';

            results.forEach(movie => {
                let posterElement = '';
                if (movie.poster && movie.poster.trim() !== '') {
                    posterElement = `<img src="${movie.poster}" class="movie-poster" alt="${movie.title}">`;
                } else {
                    posterElement = `
                        <div class="movie-poster-placeholder">
                            <div class="placeholder-icon">🎬</div>
                            <p>Постер отсутствует</p>
                        </div>`;
                }

                const movieCard = document.createElement('div');
                movieCard.className = 'movie-card';
                movieCard.innerHTML = `
                    <div class="movie-poster-container">${posterElement}</div>
                    <div class="movie-info">
                        <h3 class="movie-title">${movie.title}</h3>
                        <div class="movie-meta">
                            ${movie.rating ? `<span class="movie-rating">⭐ ${movie.rating}</span>` : '<span></span>'}
                            ${movie.genres ? `<span class="movie-genres">${movie.genres}</span>` : ''}
                        </div>
                        ${movie.age_rating ? `<div class="movie-age">${movie.age_rating}</div>` : ''}
                    </div>`;
                movieCard.addEventListener('click', () => openMovieDescription(movie));
                moviesGrid.appendChild(movieCard);
            });

            resultsContainer.style.display = 'block';
            errorContainer.style.display = 'none';
            hideLoading();
        }

        async function performSearch() {
            const query = searchInput.value.trim();
            const label = query
                ? `Запрос: "${query}"`
                : (currentFilters.genre || currentFilters.age_rating)
                    ? 'Поиск по выбранным фильтрам'
                    : 'Случайные фильмы с высоким рейтингом';
            showLoading(label);

            try {
                const response = await fetch('/search', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        query: query,
                        genre: currentFilters.genre,
                        age_rating: currentFilters.age_rating,
                    })
                });

                const rawText = await response.text();

                let data;
                try {
                    data = JSON.parse(rawText);
                } catch (parseError) {
                    console.error('Сервер вернул не JSON:', rawText);
                    throw new Error(
                        'Сервер вернул HTML вместо JSON. Обновите страницу Ctrl+F5 и проверьте, что открыт адрес http://127.0.0.1:5000/'
                    );
                }

                if (!response.ok) {
                    throw new Error(data.error || 'Произошла ошибка при поиске');
                }

                if (data.success && data.results && data.results.length > 0) {
                    showResults(data.results);
                } else {
                    showError('Ничего не найдено. Попробуйте изменить запрос или фильтры.');
                }
            } catch (error) {
                console.error(error);
                showError(error.message || 'Ошибка при поиске фильмов.');
            }
        }

        function openFilterModal() {
            filterModal.style.display = 'flex';
        }

        function closeFilterModal() {
            filterModal.style.display = 'none';
        }

        function applyFilters() {
            const selectedGenre = document.querySelector('#genreOptions .filter-option.selected');
            const selectedAge = document.querySelector('#ageOptions .filter-option.selected');

            currentFilters.genre = selectedGenre ? selectedGenre.dataset.value : null;
            currentFilters.age_rating = selectedAge ? selectedAge.dataset.value : null;

            updateActiveFiltersLabel();
            closeFilterModal();
            performSearch();
        }

        function resetFilters() {
            document.querySelectorAll('#genreOptions .filter-option, #ageOptions .filter-option').forEach(option => {
                option.classList.remove('selected');
            });
            currentFilters = { genre: null, age_rating: null };
            updateActiveFiltersLabel();
            closeFilterModal();
        }

        searchBtn.addEventListener('click', performSearch);
        searchInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') performSearch();
        });

        filterBtn.addEventListener('click', openFilterModal);
        closeMovieDescriptionBtn.addEventListener('click', closeMovieDescription);
        closeFilter.addEventListener('click', closeFilterModal);
        applyFiltersBtn.addEventListener('click', applyFilters);
        resetFiltersBtn.addEventListener('click', resetFilters);

        filterModal.addEventListener('click', (e) => {
            if (e.target === filterModal) closeFilterModal();
        });

        movieDescriptionModal.addEventListener('click', (e) => {
            if (e.target === movieDescriptionModal) closeMovieDescription();
        });

        document.querySelectorAll('.filter-option').forEach(option => {
            option.addEventListener('click', function() {
                this.parentElement.querySelectorAll('.filter-option').forEach(opt => opt.classList.remove('selected'));
                this.classList.add('selected');
            });
        });

        updateActiveFiltersLabel();
    </script>
</body>
</html>
'''

ERROR_HTML = '''<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ошибка - ФильмоПоиск</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            color: #f1f5f9;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .error-container {
            text-align: center;
            padding: 3rem;
            background: rgba(30, 41, 59, 0.9);
            border-radius: 20px;
            max-width: 600px;
            border: 1px solid rgba(244, 63, 94, 0.6);
        }
        .error-code {
            font-size: 3rem;
            font-weight: 700;
            color: #f43f5e;
        }
        .home-btn {
            display: inline-block;
            margin-top: 1rem;
            padding: 0.85rem 1.75rem;
            border-radius: 999px;
            text-decoration: none;
            color: white;
            background: linear-gradient(120deg, #4dabf7, #339af0);
        }
    </style>
</head>
<body>
    <div class="error-container">
        <div style="font-size:4rem;">⚠️</div>
        <div class="error-code">{{ error_code }}</div>
        <h1>{{ error_message }}</h1>
        <p>Проверьте наличие CSV-файлов и перезапустите приложение.</p>
        <a href="/" class="home-btn">Вернуться на главную</a>
    </div>
</body>
</html>
'''


def ensure_templates():
    os.makedirs("templates", exist_ok=True)
    with open("templates/index.html", "w", encoding="utf-8") as f:
        f.write(INDEX_HTML)
    with open("templates/error.html", "w", encoding="utf-8") as f:
        f.write(ERROR_HTML)


if __name__ == "__main__":
    ensure_templates()
    initialize_app()
    app.run(debug=True, use_reloader=False)
