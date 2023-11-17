import os

def extract_president_names(file_names):
    president_names = set()
    for file_name in file_names:
        parts = file_name.split('_')
        if len(parts) >= 3:
            president_names.add(parts[2])

    return list(president_names)

directory = "./speeches"
files_names = os.listdir(directory)
president_names = extract_president_names(files_names)
print("President Names:", president_names)
