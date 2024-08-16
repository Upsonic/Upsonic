def string_to_function(func_str):
    # Define a local dictionary to execute the function string
    local_dict = {}

    # Execute the function string in the local dictionary
    exec(func_str, globals(), local_dict)

    # Extract the function name
    func_name = func_str.split("(")[0].split()[-1]

    # Return the function object from the local dictionary
    return local_dict[func_name]
