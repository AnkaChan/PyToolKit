import logging
import sys

# logFileMode='a' or 'w', same as used in open(filename, *)

def configLogger(logFile, logFileMode='a', handlerStdoutLevel = logging.INFO, handlerFileLevel = logging.DEBUG, format=None):
    logger = logging.getLogger("logger")


    handlerStdout = logging.StreamHandler(sys.stdout)
    handlerFile = logging.FileHandler(filename=logFile, mode=logFileMode)

    logger.setLevel(logging.DEBUG)
    handlerStdout.setLevel(handlerStdoutLevel)
    handlerFile.setLevel(handlerFileLevel)

    if format is None:
        formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    else:
        formatter = logging.Formatter(format)

    handlerStdout.setFormatter(formatter)
    handlerFile.setFormatter(formatter)

    logger.addHandler(handlerStdout)
    logger.addHandler(handlerFile)

    return logger

