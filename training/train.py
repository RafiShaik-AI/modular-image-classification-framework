import tensorflow as tf
from tensorflow.keras import datasets
from models.cnn_model import create_cnn_model

(train_images,train_labels),(test_images,test_labels) = datasets.cifar10.load_data()

train_images = train_images/255.0
test_images = test_images/255.0

model = create_cnn_model()

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.fit(train_images,train_labels,epochs=5)

test_loss,test_acc = model.evaluate(test_images,test_labels)

print("Test Accuracy:",test_acc)
