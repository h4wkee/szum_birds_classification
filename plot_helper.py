import matplotlib.pyplot as plt
import pickle
import json


if __name__ == '__main__':

    # without augmentation
    # train_loss = [2.9376,2.2422,1.6798,1.2738,0.9949,0.7755,0.6566,0.5582]
    # train_accuracy = [0.3432,0.4748,0.5940,0.6850,0.7475,0.8025,0.8297,0.8564]
    #
    # validation_loss=[2.6640,3.7137,2.0378,3.4780,3.8751,3.8989,7.5565,3.3929]
    # validation_accuracy=[0.3167,0.3586,0.3665,0.3371,0.4061,0.3462,0.3688,0.3903]

    # with augmentation
    train_loss = [4.9863, 4.3811, 4.0160, 3.8167, 3.6717, 3.5717, 3.4921, 3.4533, 3.3979, 3.3690]
    validation_loss = [3.9243, 4.8975, 3.7657, 4.2727, 4.8832, 3.9490, 2.5534, 1.6197, 3.6058, 3.5735]

    train_accuracy = [0.0432, 0.1079, 0.1567, 0.1816, 0.2078, 0.2210, 0.2405, 0.2461, 0.2540, 0.2602]
    validation_accuracy = [0.1261, 0.1821, 0.2330, 0.2308, 0.2500, 0.2885, 0.3224, 0.3552, 0.2511, 0.3914]

    plt.figure(figsize=(4, 2))

    plt.subplot(121)
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.plot(range(10), train_loss, label="train")
    plt.plot(range(10), validation_loss, label="valid")
    plt.gca().legend(loc="lower left")

    plt.subplot(122)
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.plot(range(10), train_accuracy, label="train")
    plt.plot(range(10), validation_accuracy, label="valid")

    plt.show()