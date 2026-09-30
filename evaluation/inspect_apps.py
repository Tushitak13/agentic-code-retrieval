import inspect
import mteb

task = mteb.get_task("AppsRetrieval")

print("Task type:", type(task))

print("\nTask attributes:")
print([
    x for x in dir(task)
    if any(word in x.lower() for word in ["data", "corpus", "qrel", "query"])
])

print("\nload_data signature:")
print(inspect.signature(task.load_data))

print("\nevaluate signature:")
print(inspect.signature(task.evaluate))

print("\nLoading dataset...")
task.load_data()

print("\nAvailable task attributes after loading:")
print([
    x for x in dir(task)
    if any(word in x.lower() for word in ["data", "corpus", "qrel", "query"])
])

if hasattr(task, "dataset"):
    print("\nDataset:")
    print(task.dataset)

if hasattr(task, "data"):
    print("\nData:")
    print(task.data)