"""Callback Handler streams to stdout on new llm token."""

from langchain.callbacks.streaming_stdout_final_only import (
    FinalStreamingStdOutCallbackHandler,
)
from typing import Any


class customcallback(FinalStreamingStdOutCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self.append_to_last_tokens(token)

        if self.check_if_answer_reached():
            self.answer_reached = True

            return

        if self.answer_reached:
            from .gpt_computer_assistant import the_main_window

            the_main_window.set_text_to_input_box(token)
