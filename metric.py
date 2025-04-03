import json
import numpy
class Recall:
    def __init__(self, number_of_retrieved, data_path='escrcpy-commits-generated.json'):
        self.number_of_retived = number_of_retrieved
        self.data_path = data_path
        with open(self.data_path, 'r', encoding='utf-8') as file:
            self.data = json.load(file) 
        self.queries = []
        self.files = []
        for entry in self.data:
            self.queries.append(entry['question'])
            self.files.append(entry['files'])


    def single_recall(self, data, predicted_files_single):
        correct = 0
        total = len(data)
        for file in predicted_files_single:
            if file in data:
                correct += 1

        recall = correct / total
        return recall
    
    def evaluate(self, predicted_files_batch):
        print(f"Recall@{self.number_of_retived}")
        all_recall = []

        for i in range(len(predicted_files_batch)):
            all_recall.append(self.single_recall(self.files[i], predicted_files_batch[i]))

        return numpy.mean(all_recall)
    