import os

def extract_president_names(file_names):
    president_names = set()
    for file_name in file_names:
        parts = file_name.split('_')
        name = parts[1]
        if ord(name[-1]) >= 48 and ord(name[-1]) <= 57:
            name = ''
            for i in range(len(parts[1][-1])-1):
                name += str(parts[1][i])
        president_names.add(name)

    return list(president_names)

directory = "./speeches-20231110"
files_names = os.listdir(directory)
president_names = extract_president_names(files_names)
print("President Names:", president_names)
