
# copies the kg.tsv file found in the given path while skipping every nth [relation_string] relation
def modify_kg(path, n, relation_string):
    total_removed = 0
    counter = 0

    with open(f"{path}/kg.tsv", "r") as infile, open(f"{path}/every_{n}_removed_kg.tsv", "w") as outfile:
        for line in infile:
            line = line.strip()

            parts = line.split("\t")

            if len(parts) == 3 and parts[1] == relation_string:
                counter += 1

                if counter == n:
                    #print(f"skipping: {line}")
                    counter = 0
                    total_removed += 1
                    continue

            #print(f"writing: {line}")
            outfile.write(line + "\n")
            
    print(f"Removed {relation_string} total: {total_removed}")