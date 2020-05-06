# -*- coding: utf-8 -*-

import tensorflow as tf
import os
from tqdm import tqdm

from yolo.loss import loss_fn


def train_fn(model, train_generator, valid_generator, learning_rate, num_epoches, stepsPerEpoch, save_dname=None):
    save_fname = _setup(save_dname)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    history = []
    for i in range(num_epoches):

        # 1. update params
        train_loss = _train_epoch(model, optimizer, train_generator, stepsPerEpoch)

        # 2. monitor validation loss
        if valid_generator:
            valid_loss = validation_loop(model, valid_generator)
            loss_value = valid_loss
        else:
            loss_value = train_loss
        info = f"{i}-th loss = {loss_value}, train_loss = {train_loss}"
        print(info)

        # 3. update weights
        history.append(loss_value)
        if save_fname is not None and loss_value <= min(history):
            print("    update weight {}".format(loss_value))
            model.save_weights(f"{save_fname}.h5")
            with open(f"{save_fname}.txt", "wt") as f:
                f.write(info)

    return history


def _train_epoch(model, optimizer, generator, stepsPerEpoch):
    loss_value = 0
    for _ in tqdm(range(stepsPerEpoch)):
        xs, yolo_1, yolo_2, yolo_3 = generator.next_batch()
        ys = [yolo_1, yolo_2, yolo_3]
        grads, loss = _grad_fn(model, xs, ys)
        loss_value += loss
        optimizer.apply_gradients(zip(grads, model.variables))
    loss_value /= stepsPerEpoch
    return loss_value


def validation_loop(model, generator):
    loss_value = 0
    batchesCount = generator.batchesCount()
    assert batchesCount > 0
    for _ in range(batchesCount):
        xs, yolo_1, yolo_2, yolo_3 = generator.next_batch()
        ys_gt = [yolo_1, yolo_2, yolo_3]
        ys_predicted = model(xs)
        loss_value += loss_fn(ys_gt, ys_predicted)
    avgLossValue = loss_value / batchesCount
    return avgLossValue


def _setup(save_dname):
    if save_dname:
        if not os.path.exists(save_dname):
            os.makedirs(save_dname)
        save_fname = os.path.join(save_dname, "weights")
    else:
        save_fname = None
    return save_fname


def _grad_fn(model, images_tensor, list_y_trues):
    with tf.GradientTape() as tape:
        logits = model(images_tensor)
        loss = loss_fn(list_y_trues, logits)
        # print("loss = ", loss)
    return tape.gradient(loss, model.variables), loss
