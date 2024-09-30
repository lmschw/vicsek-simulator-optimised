from datetime import datetime

"""
Contains methods that can be used anywhere.
"""

def logWithTime(text):
    """
    Prints the specified text prefaced with the current date and time.

    Params:
        - text (string): the text to be printed

    Returns:
        Nothing.
    """
    dateTime = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    print(f"{dateTime}: {text}")

def formatTime(timeInSecs):
    """
    Formats seconds as hours, minutes, seconds.

    Params:
        - timeInSecs (float): the number of seconds
    
    Returns:
        A formatted string specifying the number of hours, minutes and seconds.
    """
    mins = int(timeInSecs / 60)
    secs = timeInSecs % 60

    if mins >= 60:
        hours = int(mins / 60)
        mins = mins % 60
        return f"{hours}h {mins}min {secs:.1f}s"
    return f"{mins}min {secs:.1f}s"

def createListOfFilenamesForI(baseFilename, maxI, minI=0, fileTypeString="json"):
    filenames = []
    for i in range(minI, maxI):
        filenames.append(baseFilename + f"_{i}.{fileTypeString}")
    return filenames