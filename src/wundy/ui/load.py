import logging
from typing import IO
from typing import Any

import yaml

from .schemas import input_schema

logger = logging.getLogger(__name__)


def load(file: IO[Any]) -> dict[str, dict[str, Any]]:
    data = yaml.safe_load(file)
    return input_schema.validate(data)
