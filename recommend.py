from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os
import psycopg2
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

app = Flask(__name__)

db_name = os.getenv("DB_NAME")
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
db_host = os.getenv("DB_HOST")
db_port = os.getenv("DB_PORT")

# Connect to PostgreSQL database
conn = psycopg2.connect(
    dbname=db_name,
    user=db_user,
    password=db_password,
    host=db_host
)

cursor = conn.cursor()

def get_user_views():
    query = "SELECT user_id, book_id, viewed_count FROM user_book_views"
    cursor.execute(query)
    data = cursor.fetchall()

    user_book_matrix = {}
    for row in data:
        user_id, book_id, viewed_count = row
        if user_id not in user_book_matrix:
            user_book_matrix[user_id] = {}
        user_book_matrix[user_id][book_id] = viewed_count

    return user_book_matrix

def recommend_books(user_id, user_book_matrix, min_similarity=0.0):
    if user_id not in user_book_matrix:
        return []

    # Convert the user's views into a vector
    user_vector = np.zeros(len(user_book_matrix[user_id]))
    for i, (book_id, viewed_count) in enumerate(user_book_matrix[user_id].items()):
        user_vector[i] = viewed_count

    # Calculate similarity with other users
    similarities = []
    for other_user_id, other_user_vector in user_book_matrix.items():
        if other_user_id != user_id:
            other_vector = np.zeros(len(user_book_matrix[user_id]))
            for i, (book_id, viewed_count) in enumerate(other_user_vector.items()):
                other_vector[i] = viewed_count
            similarity = cosine_similarity([user_vector], [other_vector])[0][0]
            similarities.append((other_user_id, similarity))

    # Sort users by similarity in descending order
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Get books viewed by similar users but not by the user
    recommended_books = []
    for other_user_id, similarity in similarities:
        if similarity >= min_similarity:
            for book_id, viewed_count in user_book_matrix[other_user_id].items():
                if book_id not in user_book_matrix[user_id]:
                    recommended_books.append(book_id)

    return recommended_books

@app.route('/recommend_books', methods=['GET'])
def get_recommendations():
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({"error": "User ID is required"}), 400

    user_book_matrix = get_user_views()
    recommended_books = recommend_books(int(user_id), user_book_matrix)

    return jsonify(recommended_books)

if __name__ == "__main__":
    app.run(debug=True)
