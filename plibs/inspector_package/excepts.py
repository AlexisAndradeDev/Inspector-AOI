"""Módulo con las clases de excepciones."""

class AlgorithmError(Exception):
    """Son errores que afectan a todo el proceso de inspección."""
    def __init__(self, *args):
        super().__init__(*args)

class BoardError(Exception):
    """Son errores que afectan a todo el proceso de inspección."""
    def __init__(self, *args):
        super().__init__(*args)

class FatalError(Exception):
    """Son errores que afectan a todo el proceso de inspección."""
    def __init__(self, *args):
        super().__init__(*args)
