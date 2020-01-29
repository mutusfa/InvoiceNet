from datetime import datetime
import dateutil.parser
import re

import pandas as pd
from schwifty import IBAN
import stdnum.lt


def parses_as_number(text):
    try:
        float(text)
        return True
    except ValueError:
        pass
    currencies = [r"\$", "USD", "€", "EUR"]
    currencies_pattern = "|".join(currencies)
    amount_pattern = (
        r"(?:^|\s|"
        + currencies_pattern
        + r")\d+\.\d+(?:$|\s|"
        + currencies_pattern
        + r")"
    )
    try:
        return bool(re.search(amount_pattern, text)[0])
    except TypeError:  # no matches
        return None


def parses_as_IBAN(text):
    try:
        return IBAN(text)
    except ValueError:
        return False


def parses_as_VAT_registration(text):
    return stdnum.lt.vat.is_valid(text)


def parses_as_serial_number(text):
    has_numbers = re.search(r"\d", text)
    has_letters = re.search(r"[a-zA-Z]", text)
    return has_numbers and has_letters and text.isalnum()


def parses_as_full_date(text):
    try:
        min_default = dateutil.parser.parse(
            text, default=datetime.min, ignoretz=True
        )
        max_default = dateutil.parser.parse(
            text, default=datetime.max, ignoretz=True
        )
    except (ValueError, TypeError, OverflowError):
        return False
    return min_default.date() == max_default.date()


PARSERS_DATAFRAME = pd.DataFrame(
    [
        (parses_as_full_date, 100, "date",),
        (parses_as_number, 1000, "number"),
        (parses_as_IBAN, 200, "iban"),
        (parses_as_VAT_registration, 300, "vat_registration"),
        (parses_as_serial_number, 500, "serial_number"),
    ],
    columns=["callable", "priority", "label"],
).sort_values("priority")
