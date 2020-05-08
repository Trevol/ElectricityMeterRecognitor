# -*- coding: utf-8 -*-

import tensorflow as tf
import os
from tqdm import tqdm

from yolo.loss import loss_fn


def train_fn(model, train_generator, valid_generator, learning_rate, num_epoches, stepsPerEpoch, saveDir=None,
             saveAll=False):
    saveFile = _setup(saveDir)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    history = []
    for i in range(num_epoches):
        print(f'\n{i}-th epoch...')
        # 1. update params
        train_loss = _train_epoch(model, optimizer, train_generator, stepsPerEpoch)

        # 2. monitor validation loss
        if valid_generator:
            valid_loss = validation_loop(model, valid_generator)
            loss_value = valid_loss
        else:
            loss_value = train_loss
        info = f"\n{i}-th loss = {loss_value}, train_loss = {train_loss}"
        print(info)

        # 3. update weights
        history.append(loss_value)
        if saveFile and loss_value <= min(history):
            print("  update weight {}".format(loss_value))
            model.save_weights(f"{saveFile}_{i}_{loss_value:.3f}.h5")
            with open(f"{saveFile}.txt", "wt") as f:
                f.write(info)

    return history


def _train_epoch(model, optimizer, generator, stepsPerEpoch):
    loss_value = 0
    desc = 'Training'
    avgLoss = 0.
    iterations = tqdm(generator.batches(stepsPerEpoch), desc, stepsPerEpoch)
    for i, (xs, yolo_1, yolo_2, yolo_3) in enumerate(iterations):
        ys = [yolo_1, yolo_2, yolo_3]
        grads, loss = _grad_fn(model, xs, ys)
        loss_value += loss
        optimizer.apply_gradients(zip(grads, model.variables))
        avgLoss = loss_value / (i + 1)
        iterations.set_description(f'{desc}. Curr. loss: {loss:.6f}, Avg. loss: {avgLoss:.6f}')
    # avgLoss = loss_value / stepsPerEpoch
    return avgLoss


def validation_loop(model, generator):
    loss_value = 0
    batchesCount = generator.datasetBatchesCount()
    assert batchesCount > 0
    for xs, yolo_1, yolo_2, yolo_3 in tqdm(generator.batches(), 'Validating', batchesCount):
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
