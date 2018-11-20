import os
import shutil

import torch


class Checkpoint(object):
    """Save/load checkpoints for model, optimizer parameters, and something else."""

    def __init__(self, checkpoint_dir, logger, 
                 filename='{}.pth.tar', best_checkpoint='best',
                 latest_checkpoint='latest'):
        """
        Args:
            checkpoint_dir (path): the directory of checkpoint files.
            logger (logger): the logger of application.
        """
        self.checkpoint_dir = os.getcwd()  # current working directory
        if checkpoint_dir is not None:
            self.checkpoint_dir = checkpoint_dir
        self.logger = logger
        self.filename = filename # PyTorch save/load file format
        self.latest_checkpoint = latest_checkpoint  # latest model checkpoint name
        self.best_checkpoint = best_checkpoint  # best model checkpoint name

    def freeze(self, epoch, model, optimizer=None, is_best=False,
               checkpoint=None, extra={}):
        """Save model, optimizer and other parameters to file.

        Args:
            model: the model want to save
            epoch (int): the epoch of training.
            checkpoint (str): the name of checkpoint, i.e., "latest", "best".
            is_best (bool): whether model is the best so far.
            extra (dict): other objects want to save
        """
        if not os.path.isdir(self.checkpoint_dir):
            msg = "Checkpoint Directory does not exist! Making directory {}"
            self.logger.info(msg.format(self.checkpoint_dir))
            os.makedirs(self.checkpoint_dir)
        else:
            self.logger.info("Checkpoint Directory exists!")
        if checkpoint is None:
            checkpoint = self.latest_checkpoint
        checkpoint_file = os.path.join(self.checkpoint_dir, 
                                       self.filename.format(checkpoint))
        data = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optim_state': optimizer.state_dict() if optimizer is not None else None,
            'extra': extra
        }
        # save checkpoint
        torch.save(data, checkpoint_file)
        msg = "Freeze checkpoint into file: {}"
        self.logger.info(msg.format(checkpoint_file))
        # copy best model
        if is_best and checkpoint != self.best_checkpoint:
            best_file = os.path.join(self.checkpoint_dir, 
                                     self.filename.format(self.best_checkpoint))
            shutil.copy(checkpoint_file, best_file)
            msg = "Freeze the best model checkpoint into file: {}"
            self.logger.info(msg.format(best_file))

    def restore(self, model, optimizer=None, checkpoint=None, extra={}):
        """Restore checkpoint from file.
        
        Args:
            checkpoint (str): the checkpoint name.

        Returns: `True` means success and `False` means failure.
        """
        if not os.path.isdir(self.checkpoint_dir):
            self.logger.error("Checkpoint Directory not exists! ")
            return False
        if checkpoint is None:
            checkpoint = self.best_checkpoint
        checkpoint_file = os.path.join(self.checkpoint_dir,
                                       self.filename.format(checkpoint))
        if not os.path.isfile(checkpoint_file):
            self.logger.error("Checkpoint File not exists!")
            return False
        # restore checkpoint
        data = torch.load(checkpoint_file)
        model.load_state_dict(data['model_state'])
        if optimizer is not None:
            optimizer.load_state_dict(data['optim_state'])
        msg = "Restore checkpoint from file: {}"
        self.logger.info(msg.format(checkpoint_file))
        # return other objects
        extra.update(data)
        return True
