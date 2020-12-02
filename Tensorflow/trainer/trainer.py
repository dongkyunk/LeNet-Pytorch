import tensorflow as tf
import neptune
import logging
import os


class Trainer():
    def __init__(self, train_data, valid_data, model, config):
        self.model = model
        self.train_data = train_data
        self.valid_data = valid_data
        self.config = config
        self.callbacks = list()

    def _init_callbacks(self):
        self.callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(self.config.callbacks.checkpoint_dir,
                                      '%s-{epoch:02d}-{val_loss:.2f}.hdf5' % self.config.exp.name),
                monitor=self.config.callbacks.checkpoint_monitor,
                mode=self.config.callbacks.checkpoint_mode,
                save_best_only=self.config.callbacks.checkpoint_save_best_only,
                save_weights_only=self.config.callbacks.checkpoint_save_weights_only,
                verbose=self.config.callbacks.checkpoint_verbose,
            )
        )

    def train(self):
        self.model.fit(x=self.model,
                       validation_data=self.valid_data,
                       epochs=self.config.epochs,
                       steps_per_epoch=self.config.steps,
                       validation_steps=self.config.valid_steps,
                       callbacks=self.callbacks)

    def get_loss(self, model, x, y, loss_function, training=False):
        y_ = model(x, training=training)
        return loss_function(y_true=y, y_pred=y_), y_

    @tf.function()
    def train_step(self, model, x, y, loss_function):
        with tf.GradientTape() as tape:
            loss_value, y_ = self.get_loss(
                model, x, y, loss_function, training=True)
        self._logger(loss_value)
        return loss_value, tape.gradient(loss_value, model.trainable_variables), y_

    def train_epoch(self):
        raise NotImplementedError

    def _logger(self, train_loss):
        neptune.log_metric('train_loss', train_loss)

        logging.info(
            f'train loss : {train_loss}')

