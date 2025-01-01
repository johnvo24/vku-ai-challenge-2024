import re
from datetime import datetime
import random

class Normalize():
  def normalize_number(input_str):
    """
    Convert and normalize a string containing a number into float number
    Args: 
      text(str): The input text representing a number
    Returns:
      float: The number as a float
    """
    normalized = re.sub(r'[^0-9.-BMK]', '', input_str)
    if 'M' in input_str:
      normalized = float(normalized.replace('M', '')) * 1e6
    elif 'B' in input_str:
      normalized = float(normalized.replace('B', '')) * 1e9
    elif 'K' in input_str:
      normalized = float(normalized.replace('K', '')) * 1e3
      if int(normalized) == 0: normalized = float(random.randint(1, 9))
    return normalized if normalized else float('nan')
  
  def normalize_date(date_str, format='%m/%d/%Y'):
    """
    Convert a date string in the format into an integer representing the date in '__yymmdd' format.
    Args:
      date_str (str): The input date string in the format (ex: '%d/%m/%Y').
    Returns:
      int: The date as an integer in the format '__yymmdd'.
    """
    date_str = date_str.split('/')
    if len(date_str[2]) == 2:
      date_str[2] = '20' + date_str[2][0] + date_str[2][1]
      date_str = date_str[0] + '/' + date_str[1] + '/' + date_str[2]
    else: date_str = "/".join(date_str)

    date_obj = datetime.strptime(date_str, format)
    return int(date_obj.strftime("%Y%m%d"))%1000000