class InvalidParameterException(Exception):
    """Error encountered when an invalid parameter is passed."""

    pass


class NotFoundColumnsError(Exception):
    """Error encountered when columns are not found in the dataframe."""

    pass
