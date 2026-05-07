"""Splitter abstractions and factory helpers."""

from libs.splitter.base_splitter import BaseSplitter
from libs.splitter.recursive_splitter import RecursiveSplitter, SplitterError
from libs.splitter.splitter_factory import SplitterFactory, SplitterFactoryError

SplitterFactory.register("recursive", RecursiveSplitter)

__all__ = ["BaseSplitter", "RecursiveSplitter", "SplitterError", "SplitterFactory", "SplitterFactoryError"]
