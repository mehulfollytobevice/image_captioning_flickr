def capture(seconds_elapsed):
    '''
    Function to calculate the time elapsed
    :param seconds_elapsed: number of seconds elapsed since starting
    :return: total time in the format hours:minutes:seconds
    '''
    #how many hours
    hours=int(seconds_elapsed/(60*60))
    #how many minutes
    minutes=int((seconds_elapsed%(60*60))/60)
    #how many seconds
    seconds=seconds_elapsed % 60
    return f'{hours}:{minutes:>02}:{seconds:>05.2f}'