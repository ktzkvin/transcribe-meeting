from typing import Any, List
from logging import Logger
from typing import Union
from dataclasses import asdict, is_dataclass

from src.schema.audio import AudioFileMetadata, AudioMicrophoneMetadata
from rx.core import Observer


class DebugLogger(Observer):
    def __init__(
        self,
        logger: Logger,
        write_data: bool = False,
        skip_keys: List[str] = [],
        on_next=None,
        on_error=None,
        on_completed=None,
    ):
        super().__init__(on_next, on_error, on_completed)
        self.logger = logger
        self.write_data = write_data
        self.skip_keys = skip_keys

    def on_completed(self):
        return super().on_completed()

    def on_error(self, error):
        return super().on_error(error)

    def _convert_to_dict(self, obj: Any) -> Any:
        """
        Recursively converts dataclass objects and their nested structures (lists, dicts)
        into dictionaries.
        """
        if is_dataclass(obj):
            tmp_log = asdict(obj)
            for key in self.skip_keys:
                if key in tmp_log:
                    del tmp_log[key]
            return tmp_log
        elif isinstance(obj, list):
            return [self._convert_to_dict(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self._convert_to_dict(value) for key, value in obj.items()}
        return obj

    def on_next(self, value: Any):

        log = self._convert_to_dict(value)
        if not self.write_data:
            for key in self.skip_keys:
                if key in log:
                    del log[key]
        self.logger.debug(log)


class DebugAudioMetadataLogger(DebugLogger):
    def __init__(
        self, logger, write_data=False, on_next=None, on_error=None, on_completed=None
    ):
        super().__init__(
            logger, write_data, ["audio_data"], on_next, on_error, on_completed
        )

    def on_next(self, value: Union[AudioFileMetadata, AudioMicrophoneMetadata]):
        return super().on_next(value)