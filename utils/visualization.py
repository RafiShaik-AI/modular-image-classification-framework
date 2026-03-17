import matplotlib.pyplot as plt

def plot_accuracy(history):

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])

    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')

    plt.legend(['Train','Validation'])
    plt.show()
