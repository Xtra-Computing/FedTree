filename = 'breast-cancer_scale'
# filename = 'mnist.scale'
file = open(filename, 'r')
fileout = open(filename+'clean', 'w')

labels = []
# print((len(file.readlines())))
for line in file.readlines():
    label = int(line[:2])
    if label not in labels:
        labels.append(label)

label_range = max(labels) - min(labels)
mapping = {}
for label in labels:
    mapping[label] = (label - min(labels)) * 2 / label_range - 1
print(mapping)

file=open(filename)
for line in file.readlines():
    new_label = str(mapping[int(line[:2])])
    # print(new_label)
    fileout.write(new_label + ' '+line[2:])
