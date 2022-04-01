import re
import numpy as np
import pandas as pd


# location = f'{prefix}DK{dk}+{dm}'

def dk2num(dk):
    """
    Convert Distance Kilometre to metres

    Parameters
    ----------
    dk: str
        Distance Kilometre in string format

    Returns
    -------
    float
        Distance in metres
    """
    assert re.match('^[A-Z]DK\d+\+\d{1,3}(\.\d+)?$', dk, re.IGNORECASE) is not None, 'Invalid DK format!'
    km = int(dk[3:re.match('^[A-Z]DK\d+\+', dk, re.IGNORECASE).end()-1])
    m = float(dk[re.match('^[A-Z]DK\d+\+', dk, re.IGNORECASE).end():])
    return km * 1000 + m


class Record:
    def __init__(self):
        self.table = self.initTable()

    @ staticmethod
    def initTable():
        """
        Initialize an empty table with all the columns and dtypes

        Returns
        -------
        pd.DataFrame
        """
        names = {'Project': ['Province', 'Project', 'Subproject', 'Location', 'LocationFormat'],
                 'Coordinates': ['Longitude', 'Latitude', 'Elevation'],
                 'Survey': ['Start', 'End', 'Duration', 'X', 'Y', 'Z', 'Notes', 'Surveyor'],
                 'Environment': ['Wind', 'Rain', 'Traffic', 'Machine'],
                 'Log': ['Created', 'LastModified', 'CreatedBy', 'LastModifiedBy']}
        types = {'Project': ['string'] * 4,
                 'Coordinates': ['float'] * 3,
                 'Survey': ['datetime64'] * 2 + ['float'] * 4 + ['string'] * 2,
                 'Environment': ['int'] * 4,
                 'Log': ['datetime64'] * 2 + ['string'] * 2}
        # construct dtypes
        dtypes = dict()
        for key in names:
            name0 = [(key, iname) for iname in names[key]]
            type0 = [itype for itype in types[key]]
            dtypes.update(zip(name0, type0))
        # create dataframe
        template = pd.DataFrame(columns=[np.concatenate([[ikey] * len(names[ikey]) for ikey in names.keys()]),
                                         np.concatenate([names[ikey] for ikey in names.keys()])])
        return template.astype(dtypes)

    def save(self, path):
        """
        Save the table to a csv file

        Parameters
        ----------
        path: str
            Path to the csv file
        """
        self.table.to_csv(path, index=False)  # TODO: specify output format
