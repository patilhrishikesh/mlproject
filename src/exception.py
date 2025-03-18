import sys
import logging

def error_message_detail(error, error_detail:sys):
    _,_,exc_tb = error_detail.exc_info()   #this will give us info about the error
    # this will give us the file name where the error occured
    file_name = exc_tb.tb_frame.f_code.co_filename 
    # we also have to print the error message and the line number
    error_message = "Error occured in python script name[{0}] at line number [{1}] error message [{2}]".format(
        file_name,   # this will give us the file name where the error occured
        exc_tb.tb_lineno, # this will give us the line number where the error occured
        str(error)   # we also have to print the error message
        )
    return error_message

class Custom_exception(Exception):
    #constructor for the class
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)
      
    def __str__(self):
        return self.error_message
    
