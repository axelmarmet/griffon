from typing import Any, Iterable, Iterator, TypeVar, Generic

_T = TypeVar("_T")

class tqdm(Iterator[_T], Generic[_T]):
    def __init__(self, iterable: Iterable[_T], *args: Any, **kwargs: Any) -> None: ...