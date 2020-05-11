import utils.suppressTfWarnings
from counter_screen.model.CounterScreenModel import CounterScreenModel
from utils.MultiDirectoryBatchGenerator import MultiDirectoryBatchGenerator
from yolo.train import train_fn
import utils.augmentations as augmentations


class CounterScreenModelTraining:
    def __init__(self, model,
                 trainDataDirs,
                 valDataDirs,
                 batch_size=8,
                 learning_rate=.0001,
                 augmentations=None,
                 saveDir='./weights/',
                 n_epoches=30,
                 stepsPerEpoch=1000):
        self.model: CounterScreenModel = model
        self.stepsPerEpoch = stepsPerEpoch
        self.n_epoches = n_epoches
        self.saveDir = saveDir
        self.augmentations = augmentations
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.valDataDirs = valDataDirs
        self.trainDataDirs = trainDataDirs

    def _trainValGenerators(self):
        trainGen = MultiDirectoryBatchGenerator(self.trainDataDirs, labelNames=self.model.labelNames,
                                                batch_size=self.batch_size, anchors=self.model.anchors,
                                                image_size=self.model.net_size, shuffleData=True,
                                                augmentations=augmentations.make())
        valGen = MultiDirectoryBatchGenerator(self.trainDataDirs, labelNames=self.model.labelNames,
                                              batch_size=self.batch_size, anchors=self.model.anchors,
                                              image_size=self.model.net_size, shuffleData=False,
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
    model = CounterScreenModel('weights/2_from_scratch/weights.h5')
    trainParams = dict(
        trainDataDirs=[
            # './dataset/counters/0_from_internet/train',
            './dataset/counters/1_from_phone/train',
            './dataset/counters/2_from_phone/train'
        ],
        valDataDirs=[
            # './dataset/counters/0_from_internet/val',
            './dataset/counters/1_from_phone/val',
            './dataset/counters/2_from_phone/val'
        ],
        batch_size=8,
        learning_rate=.0001,
        augmentations=augmentations.make(),
        saveDir='./weights/',
        n_epoches=30,
        stepsPerEpoch=1000)

    training = CounterScreenModelTraining(model, **trainParams)
    training()


if __name__ == '__main__':
    main()
