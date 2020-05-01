# -*- coding: utf-8 -*-

''' This doesn't seem right... '''

import logging
import logging.config


def configure_logger(name, log_path=None):
    logging.config.dictConfig({
        'version': 1,
        'formatters': {
            'default': {'format': '%(asctime)s - %(module)s - %(lineno)d - %(levelname)-10s - %(message)s'},
            'consolef': {'format': '%(module)s - %(lineno)d - %(levelname)-10s - %(message)s'}
        },
        'handlers': {
            'console': {
                'level': 'DEBUG',
                'class': 'logging.StreamHandler',
                'formatter': 'consolef',
                'stream': 'ext://sys.stdout'
            },
            'file': {
                'level': 'DEBUG',
                'class': 'logging.handlers.RotatingFileHandler',
                'formatter': 'default',
                'filename': log_path
            }
        },
        'loggers': {
            'default': {
                'level': 'DEBUG',
                'handlers': ['console', 'file']
            }
        },
        'disable_existing_loggers': False
    })
    return logging.getLogger(name)


def remove_console_handler(logger):
    for i, handler in enumerate(logger.handlers):
        if isinstance(handler, logging.StreamHandler):
            break

    del logger.handlers[i]
    return logger