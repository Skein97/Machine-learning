from imageai.Classification import ImageClassification
import os

exec_path = os.getcwd()

prediction = ImageClassification()

prediction.setModelTypeAsDenseNet121()
prediction.setModelPath(os.path.join(exec_path, 'densenet121-a639ec97.pth'))
prediction.loadModel()

predictions, probabilities = prediction.classifyImage(os.path.join(exec_path, 'house.jpg'), result_count=5)
for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(f'{eachPrediction} : {eachProbability}')