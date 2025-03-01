import os
import re
import sys
import json
import logging

from typing import Type, Any, Optional, Union, List
from config.consts import Consts as _C


T = _C.T


class Singleton:
    """
    A non-thread-safe helper class to ease implementing singletons.
    This should be used as a decorator -- not a metaclass -- to the
    class that should be a singleton.
    The decorated class can define one `__init__` function that
    takes only the `self` argument. Also, the decorated class cannot be
    inherited from. Other than that, there are no restrictions that apply
    to the decorated class.
    To get the singleton instance, use the `instance` method. Trying
    to use `__call__` will result in a `TypeError` being raised.
    SeeAlso:
        https://stackoverflow.com/a/7346105
        https://dev.to/vtsen/how-to-create-singleton-class-in-kotlin-5123
    """

    def __init__(self, decorated):
        self._decorated = decorated

    def instance(self):
        """
        Returns the singleton instance. Upon its first call, it creates a
        new instance of the decorated class and calls its `__init__` method.
        On all subsequent calls, the already created instance is returned.
        """
        try:
            return self._instance
        except AttributeError:
            self._instance = self._decorated()
            return self._instance

    def __call__(self):
        raise TypeError('Singletons must be accessed through `instance()`.')

    def __instancecheck__(self, inst):
        return isinstance(inst, self._decorated)


class Utils:

    @staticmethod
    def str2bool(s: str) -> bool:
        return s.lower() in ("true", "yes", "y")

    @staticmethod
    def str2lst(s: str) -> List[Any]:
        return re.findall(r"[\w']+", s)

    @staticmethod
    def check_string(s: str, options: List[str], case_sensitive: bool, exact_match: bool) -> bool:
        if case_sensitive:
            if exact_match:
                return any(map(lambda x: s == x, options))
            else:
                return any(map(lambda x: s in x, options))
        else:
            if exact_match:
                return any(map(lambda x: s.lower() == x.lower(), options))
            else:
                return any(map(lambda x: s.lower() in x.lower(), options))

    @staticmethod
    def validate_path(s: str) -> str:
        if not s:
            raise ValueError(f"Empty path.")
        
        path = os.path.abspath(os.path.realpath(s))
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path '{path}' does not exist.")

        return path

    @staticmethod
    def read_json(str_path: str) -> Any:
        """Validate json path and return instance of dict
        
        Args:
            str_path (str): the string path containing the json file
        Returns:
            obj (Any): The return type is Any because json.loads actually returns Any
        
        """
        
        json_path = None
        try:
            json_path = Utils.validate_path(str_path)
        except FileNotFoundError as fnf:
            Logger.instance().critical(f"{fnf.args} Quitting...")
            sys.exit(-1)
        finally:
            if (json_path is None):
                Logger.instance().critical("json path is null")
                sys.exit(-1)

        with open(json_path, "r") as f:
            obj = None
            try:
                obj = json.loads(f.read().replace("\n", ""))
                Utils.check_instance(obj, dict, error_json=True)
            except TypeError as te:
                Logger.instance().critical(f"{te.args}")
                sys.exit(-1)
            return obj

    @staticmethod
    def check_instance(val: Any, class_type: Optional[Type[T]], **kwargs) -> bool:
        arg_json = "Provide json string" if kwargs.get("error_json", None) else None

        if class_type is None:
            if val is not None:
                raise TypeError(f"{val} must be None.")
            else:
                return True

        if isinstance(val, class_type):
            return True
        else:
            raise TypeError(f"{val} must be {class_type.__name__}.", arg_json)

    @staticmethod
    def check_bool(val: Union[str, bool]) -> bool:
        return val if isinstance(val, bool) else Utils.str2bool(val)
    
    @staticmethod
    def invert_dict(in_dict: dict) -> dict:
        if not len(list(in_dict.values())) == len(set(in_dict.values())):
            raise ValueError("The dictionary cannot be inverted as its values are not unique")

        return { v: k for k, v in in_dict.items() }


@Singleton
class Logger:

    def __init__(self, formatter='%(asctime)-2s # %(levelname)-2s # %(message)s'):
        formatter = logging.Formatter(formatter)
        log_file_path = os.path.join(os.getcwd(), "log.log")

        # logger basic config
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        # handlers
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(formatter)

        out_handler = logging.StreamHandler(sys.stdout)
        out_handler.setLevel(logging.DEBUG)
        out_handler.addFilter(lambda record: record.levelno == logging.DEBUG)
        out_handler.setFormatter(formatter)

        err_handler = logging.StreamHandler(sys.stderr)
        err_handler.setLevel(logging.WARNING)
        err_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(out_handler)
        self.logger.addHandler(err_handler)

    def info(self, msg: str) -> None:
        self.logger.info(msg)

    def debug(self, msg: str) -> None:
        self.logger.debug(msg)

    def warning(self, msg: str) -> None:
        self.logger.warning(msg)

    def error(self, msg: str) -> None:
        self.logger.error(msg)

    def critical(self, msg: str) -> None:
        self.logger.critical(msg)