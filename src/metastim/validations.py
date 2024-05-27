import sys

def validate_electrode_list(electrode_list):
    """Validate electrode configuration exists if any electrode value is not one of -1 , 0  or 1
       Args:
        electrode_list: electrode configuration list 
       Returns:
        None 
    """
    for elec in electrode_list:
        if elec not in [-1, 0, 1]:
            print("Invalid electrode configuration. Elements must be either -1, 0, or 1.")
            sys.exit(4)