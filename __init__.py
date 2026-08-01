import sys as _sys

from . import bertalign as _impl
from .bertalign import Bertalign
from .bertalign import aligner, corelib, encoder, eval, utils

for _name in ('aligner', 'corelib', 'encoder', 'eval', 'utils'):
    _sys.modules[f'{__name__}.{_name}'] = getattr(_impl, _name)
