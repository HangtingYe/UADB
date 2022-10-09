import logging

from pipeline import Pipeline
from config import get_config_from_command


if __name__ == '__main__':
    config = get_config_from_command()

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO,
    )

    pipeline = Pipeline(config)
    pipeline.boost_co_train()