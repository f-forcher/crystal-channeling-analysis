import readline as rd

# https://stackoverflow.com/questions/2533120/show-default-value-for-editing-on-python-input-possible
def edit_input(prompt, prefill=''):
   """
   Function that writes on the cli an editable text and then awaits for input,
   useful to provide a default value as possible input.

   prompt: The same as for input(), the non editable part of the text written to
           screen.
   prefill: the editable text written to the screen.

   return: the user input as a string.
   """
   rd.set_startup_hook(lambda: rd.insert_text( str(prefill) ))
   try:
      return input(prompt)
   finally:
      rd.set_startup_hook()
