import tensorflow as tf
import numpy as np
import time
import tqdm
from .Vis import *
from os.path import join

class CNNConfig():
    def __init__(s):
        s.networkName = None

        s.lrDecayStep = None
        s.lrDecayRate = None
        s.momentum = None

        s.learningRate = None
        s.weight_decay = None

        s.batchSize = None
        s.numEpoch = None
        s.printIEpochStep = None
        s.saveStep = None
        s.plotStep = None

        s.infoOutFolder=None
        s.ckpSaveFolder=None


class CNNClassifier():
    def __init__(s, inputShape=None, labelShape=None, cfg=CNNConfig(), resetDefaultGraph=True):
        if resetDefaultGraph:
            tf.reset_default_graph()

        s.cfg = cfg

        s.inputsShape = inputShape
        s.labelsShape = labelShape

        s.inputs = tf.placeholder(tf.float32, shape=[None, *s.inputsShape], name='inputs')
        s.labels = tf.placeholder(tf.float32, shape=[None, *s.labelsShape], name='labels')
        s.is_training = tf.placeholder(tf.bool, None, name='is_training')
        s.learnrate_ph = tf.placeholder(tf.float32, name="learnrate_ph")
        s.lrDecayStepPH = tf.placeholder(tf.float32, name="lrDecayStePH")
        s.lrDecayRatePH = tf.placeholder(tf.float32, name="lrDecayRatePH")

        s.weight_decay = tf.placeholder(tf.float32, None, name='weight_decay')
        s.regularizer = tf.contrib.layers.l2_regularizer(scale=s.weight_decay)

        s.logits = None
        s.softmaxes = None
        s.loss = None
        s.optimizer = None

    def getNetwork(s):
        return "logits", "softmax"

    def getLoss(s):
        return "loss"

    def getOptimizer(s):
        s.rate=None
        return "Optimizer"

    def initialize(s):
        s.logits, s.softmaxes = s.getNetwork()
        s.loss = s.getLoss()
        s.optimizer = s.getOptimizer()

        s.sess = tf.Session()
        s.saver = tf.train.Saver()

        s.init = tf.global_variables_initializer()
        s.sess.run(s.init)

        s.actualIEpoch=0

        s.trainInfo = {
            'trainLoss':[],
            'trainAcc':[],
            'testAcc':[],
            'IEpoch':[]
        }

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

            sfmx1a = sess.run(self.softmaxes, feed_dict=feedDict)
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

    def getSaveName(s, iEpoch):
        return "{:s}_Epoch_{:8d}.ckpt".format(s.cfg.networkName, iEpoch)

    def saveCheckPoint(s, saveFileName):
        save_path = s.saver.save(s.sess, saveFileName)
        print("Model saved in path: %s" % save_path)

    def getInfoStr(s, epochNum, trainAcc, trainLoss, testAcc, learningRate):
        info = "Epoch: {:3d}, Train Acc: {:f}, Train Loss: {:f}, Test Acc: {:f}, LR: {:f}, time {:05.2f}".format(
            epochNum, trainAcc, trainLoss, testAcc, learningRate,
            time.clock() - s.timeStart
        )

        return info

    def train(s, imgsTrain, labelTrain, imgsTest, labelTest, showTqdmBar=False, updateFrequency=0.1, useActualNumEpoch=False):
        s.timeStart = time.clock()
        learningRateDecay = s.cfg.learningRate
        loss = -1

        sizeTrain = len(imgsTrain)
        numBatchs = int(np.ceil(sizeTrain / s.cfg.batchSize))

        for iEpoch in range(0, s.cfg.numEpoch):
            # np.random.shuffle(indices)
            if showTqdmBar:
                t = tqdm.trange(numBatchs, desc="Epoch_{:05d}_Loss:{:6f}_LR{:.8f}:".format(iEpoch, loss, learningRateDecay), leave=True)
            else:
                t = range(numBatchs)
            for iBatch in t:
                batchImgs = imgsTrain[iBatch * s.cfg.batchSize:(iBatch + 1) *  s.cfg.batchSize, :, ]
                batchLabels = labelTrain[iBatch * s.cfg.batchSize:(iBatch + 1) *  s.cfg.batchSize, :]

                trainDict = s.runTrainStep(batchImgs, batchLabels)

                if showTqdmBar and  np.random.rand() < updateFrequency:
                    loss = s.sess.run(s.loss, trainDict)
                    learningRateDecay = s.sess.run(s.rate, feed_dict=trainDict)
                    t.set_description("Epoch_{:05d}_Loss:{:6f}_LR{:e}:".format(iEpoch, loss, s.sess.run(s.rate, feed_dict=trainDict)), refresh=True)
            s.actualIEpoch += 1

            if useActualNumEpoch:
                epochNum = s.actualIEpoch
            else:
                epochNum = iEpoch

            if not epochNum % s.cfg.printIEpochStep:

                predictionsTrain, trainAcc, trainLoss = s.evaluate(s.sess, imgsTrain, labelTrain,
                                                                     batchSize=s.cfg.batchSize, calculateMeanLoss=True, )
                # for quick debugging predictionsTrain, trainAcc, trainLoss = cnn.evaluate(sess, imgsTrain[:batchSize,...], labelTrain[:batchSize,...], batchSize=batchSize, calculateMeanLoss=True)
                predictionsTest, testAcc = s.evaluate(s.sess, imgsTest, labelTest, batchSize=s.cfg.batchSize)

                infoStr = s.getInfoStr(epochNum, trainAcc, trainLoss, testAcc, s.sess.run(s.rate, feed_dict=trainDict))
                print(infoStr)

                s.trainInfo['trainLoss'].append(trainLoss)
                s.trainInfo['trainAcc'].append(trainAcc)
                s.trainInfo['testAcc'].append(testAcc)
                s.trainInfo['IEpoch'].append(epochNum)

            if not epochNum % s.cfg.plotStep:
                predictionsTest, testAcc = s.evaluate(s.sess, imgsTest, labelTest, batchSize=s.cfg.batchSize)
                predictionsTestText = s.labelsToText(predictionsTest,)
                visualizeImgsAndLabel(imgsTest, predictionsTestText,
                                      outFile=join(s.cfg.infoOutFolder, 'PredTest_Epoch{:05d}.pdf'.format(epochNum)),
                                      closeExistingFigures=True)

                predictionsTrain, trainAcc, trainLoss = s.evaluate(s.sess, imgsTrain, labelTrain,
                                                                     batchSize=s.cfg.batchSize, calculateMeanLoss=True, )
                predictionsTrainText = s.labelsToText(predictionsTrain, )
                visualizeImgsAndLabel(imgsTrain, predictionsTrainText,
                                      outFile=join(s.cfg.infoOutFolder, 'PredTrain_Epoch{:05d}.pdf'.format(epochNum)),
                                      closeExistingFigures=True)

                drawErrCurves(s.trainInfo['IEpoch'], s.trainInfo['trainLoss'], xlabel='Epoch', ylabel='Train Loss', saveFile=join(s.cfg.infoOutFolder, 'TrainLoss.pdf'))
                drawErrCurves(s.trainInfo['IEpoch'], s.trainInfo['trainAcc'], xlabel='Epoch', ylabel='Train Accuracy', saveFile=join(s.cfg.infoOutFolder, 'TrainAccuracy.pdf'))
                drawErrCurves(s.trainInfo['IEpoch'], s.trainInfo['testAcc'], xlabel='Epoch', ylabel='Test Accuracy', saveFile=join(s.cfg.infoOutFolder, 'TestAccuracy.pdf'))

            if not epochNum % s.cfg.saveStep and epochNum:
                outFile = join(s.cfg.ckpSaveFolder, s.getSaveName(epochNum))
                s.saveCheckPoint(outFile)


    def runTrainStep(s, batchImgs, batchLabels):
        trainDict = {
            s.inputs: batchImgs,
            s.labels: batchLabels,
            s.learnrate_ph: s.cfg.learningRate,
            s.weight_decay: s.cfg.weight_decay,
            s.lrDecayRatePH: s.cfg.lrDecayRate,
            s.lrDecayStepPH: s.cfg.lrDecayStep,
            s.is_training: True
        }

        s.sess.run(s.optimizer, feed_dict=trainDict)

        return trainDict

    def labelsToText(s, labels):
        return "Convert labels to string"