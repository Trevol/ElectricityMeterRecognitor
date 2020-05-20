import utils.suppressTfWarnings
from digits_on_screen import DigitsOnScreenModel
from digits_on_screen.DigitsOnScreenModel import DigitsOnScreenModel
from digits_on_screen.dataset.generator import NumberImageGenerator
from yolo.train import train_fn
import utils.augmentations as augmentations


class DigitsOnScreenModelTraining:
    def __init__(self, model,
                 datasetDir,
                 batch_size=8,
                 learning_rate=.0001,
                 augmentations=None,
                 saveDir='./weights/',
                 n_epoches=30,
                 stepsPerEpoch=1000):
        self.model: CounterScreenModel = model
        self.datasetDir = datasetDir
        self.stepsPerEpoch = stepsPerEpoch
        self.n_epoches = n_epoches
        self.saveDir = saveDir
        self.augmentations = augmentations
        self.learning_rate = learning_rate
        self.batch_size = batch_size

    def _trainValGenerators(self):
        trainGen = NumberImageGenerator(datasetDir=self.datasetDir,
                                        batchSize=self.batch_size,
                                        netSize=self.model.net_size,
                                        anchors=self.model.anchors,
                                        augmentations=self.augmentations)
        valGen = NumberImageGenerator(datasetDir=self.datasetDir,
                                      batchSize=self.batch_size,
                                      netSize=self.model.net_size,
                                      anchors=self.model.anchors,
                                      augmentations=None)

        return trainGen, valGen

    def train(self):
        train_generator, valid_generator = self._trainValGenerators()
        train_fn(self.model.yolonet(),
                 train_generator,
                 valid_generator,
                 learning_rate=self.learning_rate,
                 saveDir=self.saveDir,
                 num_epoches=self.n_epoches,
                 stepsPerEpoch=self.stepsPerEpoch)

    def __call__(self, *args, **kwargs):
        self.train()


def main():
    model = DigitsOnScreenModel('./weights/6/weights_10_4.348.h5')
    trainParams = dict(
        datasetDir='./dataset/28x28',
        batch_size=8,
        learning_rate=.00001,
        augmentations=augmentations.make(1),
        saveDir='./weights/',
        n_epoches=30,
        stepsPerEpoch=1000)

    training = DigitsOnScreenModelTraining(model, **trainParams)
    training()


if __name__ == '__main__':
    main()
