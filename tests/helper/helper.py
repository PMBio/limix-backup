def message(RV):
    OK   = '\033[92m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    if RV:
        out=OK+"ok"+ENDC
    else:
        out=FAIL+"failed"+ENDC
    return out