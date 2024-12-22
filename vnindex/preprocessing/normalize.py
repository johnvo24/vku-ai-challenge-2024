import re
from datetime import datetime

class Normalize():
  def normalize_number(input_str):
    """
    Convert and normalize a string containing a number into float number
    Args: 
      text(str): The input text representing a number
    Returns:
      float: The number as a float
    """
    return float(re.sub(r'[^0-9.-]', '', input_str))
  def normalize_date(date_str, format='%d/%m/%Y'):
    """
    Convert a date string in the format into an integer representing the date in '__yymmdd' format.
    Args:
      date_str (str): The input date string in the format (ex: '%d/%m/%Y').
    Returns:
      int: The date as an integer in the format '__yymmdd'.
    """
    date_obj = datetime.strptime(date_str, format)
    return int(date_obj.strftime("%Y%m%d"))%1000000
