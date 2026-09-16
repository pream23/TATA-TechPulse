import argparse
import tensorflow as tf
from tensorflow.keras import layers, models

def build_model(num_classes):
    return models.Sequential([
        layers.Input((64,64,3)),
        layers.Rescaling(1./255),
        layers.Conv2D(32,3,activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(64,3,activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(128,3,activation="relu"), layers.MaxPooling2D(),
        layers.Flatten(), layers.Dropout(.35),
        layers.Dense(128,activation="relu"),
        layers.Dense(num_classes,activation="softmax")
    ])

def main(path, epochs=5):
    ds=tf.keras.utils.image_dataset_from_directory(path,image_size=(64,64),batch_size=32,
                                                   validation_split=.2,subset="both",seed=42)
    train,val=ds
    model=build_model(len(train.class_names))
    model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
    model.summary()
    model.fit(train,validation_data=val,epochs=epochs)
    model.save("traffic_sign_cnn.keras")

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--data",required=True);p.add_argument("--epochs",type=int,default=5)
    a=p.parse_args();main(a.data,a.epochs)
