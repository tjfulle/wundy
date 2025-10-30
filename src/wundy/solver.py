from typing import Any


class Solver:
    type: str = "<none>"
    subclasses: list["Solver"] = []

    def __init__(self, **options: Any) -> None: ...

    def __init_subclass__(cls):
        Solver.subclasses.append(cls)

    @classmethod
    def factory(cls, type: str, options: dict[str, float]) -> "Solver":
        for subclass in cls.subclasses:
            if subclass.type.upper() == type.upper():
                return subclass(**options)
        raise TypeError(f"Unknown solver type {type!r}")
