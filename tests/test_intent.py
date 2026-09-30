from src.query.intent import classify_query


queries = [
    "How is user input processed?",
    "Where is Bluetooth settings deeplink used?",
    "Which functions call authenticate?",
    "Which function runs before getUserProfile?",
]


for query in queries:

    result = classify_query(query)

    print("\nQUERY:")
    print(query)

    print("INTENT:")
    print(result)