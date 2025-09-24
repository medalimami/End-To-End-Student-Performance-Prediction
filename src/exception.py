import sys

def error_message_detail(error, error_detail: sys):
    """
    Returns a detailed error message including the file name,
    line number, and original error message.
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_message = f"Error occurred in Python script [{file_name}] line [{line_number}] error message [{error}]"
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        # Initialize the base Exception with the original message
        super().__init__(error_message)
        # Create a detailed error message
        self.error_message = error_message_detail(error_message, error_detail)
        self.error_detail = error_detail

    def __str__(self):
        return self.error_message
