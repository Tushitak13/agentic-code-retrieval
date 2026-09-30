import mteb
from sentence_transformers import SentenceTransformer


model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

task = mteb.get_task("AppsRetrieval")

result = mteb.evaluate(
    model,
    [task],
    encode_kwargs={"batch_size": 64},
)

task_result = list(result.task_results)[0]

print(task_result)