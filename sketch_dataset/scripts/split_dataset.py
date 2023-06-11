import json
import random
import sys

from sklearn.model_selection import train_test_split

index_file = sys.argv[1]
data_list = json.load(open(index_file))
random.shuffle(data_list)
train_data, test_data = train_test_split(data_list, test_size=0.3)
json.dump(train_data, open(index_file.replace('.json', '_train.json'), 'w'))
json.dump(test_data, open(index_file.replace('.json', '_test.json'), 'w'))
