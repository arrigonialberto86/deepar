import logging.config
import os

LOG_CONF = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'simple': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        }
    },
    'handlers': {
        'stream': {
            'class': 'logging.StreamHandler',
            'level': 'DEBUG',
            'formatter': 'simple',
            'stream': 'ext://sys.stdout',
        },
    },
    'loggers': {
        'deepar': {
            'handlers': ['stream'],
            'level': os.getenv('DF_LOG_LEVEL', 'DEBUG'),
        }
    },
}

logging.config.dictConfig(LOG_CONF)