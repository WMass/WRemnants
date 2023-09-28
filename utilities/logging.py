import os, logging
import time

class CustomFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors"""

    green = "\x1b[1;32m"
    grey = "\x1b[38;21m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    #format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    myformat = "%(levelname)s:%(filename)s: %(message)s"

    FORMATS = {
        logging.DEBUG: green + myformat + reset,
        logging.INFO: grey + myformat + reset,
        logging.WARNING: yellow + myformat + reset,
        logging.ERROR: red + myformat + reset,
        logging.CRITICAL: bold_red + myformat + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

logging_verboseLevel = [logging.CRITICAL, logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG]

def set_logging_level(log, verbosity):
    log.setLevel(logging_verboseLevel[max(0, min(4, verbosity))])

def setup_logger(basefile, verbosity=3, no_colors=False, initName="wremnants"):

    setup_func = setup_base_logger if no_colors else setup_color_logger
    logger = setup_func(os.path.basename(basefile), verbosity, initName)
    # count messages of base logger
    base_logger = logging.getLogger("wremnants")
    add_logging_counter(base_logger)
    # stop total time
    add_time_info("Total time")
    return logger

def setup_color_logger(name, verbosity, initName="wremnants"):
    base_logger = logging.getLogger(initName)
    # set console handler
    ch = logging.StreamHandler()
    ch.setFormatter(CustomFormatter())
    base_logger.addHandler(ch)
    set_logging_level(base_logger, verbosity)
    base_logger.propagate = False # to avoid propagating back to root logger, which would print messages twice
    return base_logger.getChild(name)
    
def setup_base_logger(name, verbosity, initName="wremnants"):
    logging.basicConfig(format='%(levelname)s: %(message)s')
    base_logger = logging.getLogger(initName)
    set_logging_level(base_logger, verbosity)
    return base_logger.getChild(name)
    
def child_logger(name, initName="wremnants"):
    # count messages of child logger
    logger = logging.getLogger(initName).getChild(name)
    add_logging_counter(logger)
    return logger

class LoggingCounterHandler(logging.Handler):
    def __init__(self, level):
        super().__init__()
        self.count = 0
        self.level = level

    def emit(self, record):
        if record.levelno == self.level:
            self.count += 1

def add_logging_counter(logger, levels=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]):
    logger.counter = {}
    for level in levels:
        logger.counter[level] = LoggingCounterHandler(getattr(logging,level))
        logger.addHandler(logger.counter[level])

def print_logging_count(logger, verbosity=logging.WARNING):
    if hasattr(logger, "counter"):
        for level, counter in logger.counter.items():
            if getattr(logging, level) < verbosity:
                continue
            if counter.count > 0:
                getattr(logger, level.lower())(f"Logger {logger.name} triggered {level} messages: {counter.count}")

def add_time_info(tag, logger=logging.getLogger("wremnants")):
    if not hasattr(logger, "times"):
        logger.times = {}
    logger.times[tag] = time.time()

def print_time_info(logger):
    if hasattr(logger, "times"):
        time_end = time.time()
        logger.info(f"Time tags summary {logger.name}:")
        for tag, itime in logger.times.items():
            logger.info(f"{tag}: {time_end - itime}")

def summary(verbosity=logging.WARNING, extended=True):
    base_logger = logging.getLogger("wremnants")

    base_logger.info(f"--------------------------------------")
    base_logger.info(f"----------- logger summary -----------")
    base_logger.info(f"--------------------------------------")

    print_time_info(base_logger)

    print_logging_count(base_logger, verbosity=verbosity)

    if not extended:
        return

    # Iterate through all child loggers and print their names, levels, and counts
    all_loggers = logging.Logger.manager.loggerDict
    for logger_name, logger_obj in all_loggers.items():
        if logger_name.startswith('wremnants.'):
            print_logging_count(logger_obj, verbosity=verbosity)



