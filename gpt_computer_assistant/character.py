name_ = "GPT Computer Assistant"
def name():
    global name_
    return name_

def change_name(new_name):
    global name_
    name_ = new_name

    from .gpt_computer_assistant import the_main_window

    def adjust_string_length(input_string):

        if len(input_string) < 18:

            return input_string.ljust(18)
        else:
   
            return input_string[:18]

    the_main_window.title_label.setText(adjust_string_length(name_))



developer_ = "Open Source Community"
def developer():
    global developer_
    return developer_


def change_developer(new_developer):
    global developer_
    developer_ = new_developer