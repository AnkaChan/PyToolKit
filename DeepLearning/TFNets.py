import tensorflow as tf
import numpy as np

class CNNConfig():
    def __init__(s):
        s.lrDecayStep = None
        s.lrDecayRate = None
        s.momentum = None

        s.learningRate = None
        s.weight_decay = None

        s.numEpoch = None
        s.printIEpochStep = None
        s.saveStep = None

class CNNClassifier():
    def __init__(s, inputShape=None, labelShape=None, cfg=CNNConfig()):
        s.cfg = cfg

        s.inputsShape = inputShape
        s.labelsShape = labelShape

        s.inputs = tf.placeholder(tf.float32, shape=[-1, *s.inputsShape], name='inputs')
        s.labels = tf.placeholder(tf.float32, shape=[-1, *s.labelsShape], name='labels')
        s.is_training = tf.placeholder(tf.bool, None, name='is_training')
        s.learnrate_ph = tf.placeholder(tf.float32, name="learnrate_ph")
        s.lrDecayStepPH = tf.placeholder(tf.float32, name="lrDecayStePH")
        s.lrDecayRatePH = tf.placeholder(tf.float32, name="lrDecayRatePH")

        s.logits = None
        s.softmax1a = None
        s.loss = None
        s.optimizer = None

    def getNetwork(s):
        return "logits", "softmax"

    def getLoss(s):
        return "loss"

    def getOptimizer(s):
        return "Optimizer"

    def initialize(s, resetDefaultGraph=True):
        if resetDefaultGraph:
            tf.reset_default_graph()

        s.logits, s.softmax1a = s.getNetwork()
        s.loss = s.getLoss()
        s.optimizer = s.getOptimizer()

    def predict(self, sess, imgs, batchSize = 200, calculateMeanLoss=False, labels=None):
        numBatchs = int(np.ceil(imgs.shape[0] / batchSize))

        totalLoss = 0
        predictionsAll = []
        for iBatch in range(numBatchs):
            batchData = imgs[iBatch * batchSize:(iBatch + 1) * batchSize, :, ... ]
            feedDict = {
                self.inputs: batchData,
                self.is_training: False,
            }

            sfmx1a = sess.run(self.softmax1a, feed_dict=feedDict)
            predict1a = np.argmax(sfmx1a, axis=1)
            predictionsAll.append(predict1a)

            if calculateMeanLoss:
                feedDict[self.labels] = labels[iBatch * batchSize:(iBatch + 1) * batchSize, :, ]
                totalLoss = totalLoss + sess.run(self.loss, feed_dict=feedDict) * batchData.shape[0]

        predictionsAll = np.concatenate(predictionsAll, axis=0)

        if calculateMeanLoss:
            return predictionsAll, totalLoss/imgs.shape[0]
        else:
            return predictionsAll

    def evaluate(self, sess, imgs, labels, batchSize=200, calculateMeanLoss=False):

        if calculateMeanLoss:
            predictionsAll, meanLoss = self.predict(sess, imgs, batchSize, calculateMeanLoss=calculateMeanLoss, labels=labels)

            accuracy = np.count_nonzero(predictionsAll == np.argmax(labels, axis=1)) / predictionsAll.shape[0]
            return predictionsAll, accuracy, meanLoss
        else:
            predictionsAll = self.predict(sess, imgs, batchSize)

            accuracy = np.count_nonzero(predictionsAll==np.argmax(labels, axis=1)) / predictionsAll.shape[0]
            return predictionsAll, accuracy
