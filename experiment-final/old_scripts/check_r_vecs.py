from load_vectors import load_person_vectors

person_vectors = load_person_vectors(
    run_path="./100people/runs/every_2_removed",
    original_index=None,
    relation_filter="hasAge",
    head_prefix="person",
)

print(type(person_vectors))
print(person_vectors.keys())

for key, item in person_vectors.items():
    query_vec = item["query_vec"]
    print(key, query_vec)