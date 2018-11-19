import logging


class Logger(object):
    """A wrapper class of get/set logger."""

    @staticmethod
    def get(name=None):
        """Get the logger by name.
        
        Args:
            name (str): the name of logger you want get. the default 
            is the root logger.

        Returns: the logger you want it.

        Note: Maybe you should invoke set_logger() before get it.
        """
        return logging.getLogger(name)

    @staticmethod
    def set(log_file, name=None, level=logging.INFO):
        """Set the logger to log info in console and file `log_file`.

        Args:
            log_file (path): the path of log file.
            name (str): logger name, `None` means the root logger.

        Returns: the logger.
        """
        logger = logging.getLogger(name)
        logger.setLevel(level)
        # just add handlers for logger only once
        if not logger.hasHandlers():
            # Logging to a file
            fh = logging.FileHandler(log_file)
            fh.setLevel(logging.DEBUG)
            fmt = logging.Formatter('%(asctime)s:%(levelname)s: %(message)s')
            fh.setFormatter(fmt)
            logger.addHandler(fh)
            # Logging to console
            ch = logging.StreamHandler()
            ch.setLevel(logging.ERROR)
            ch.setFormatter(logging.Formatter('%(message)s'))
            logger.addHandler(ch)
        return logger
