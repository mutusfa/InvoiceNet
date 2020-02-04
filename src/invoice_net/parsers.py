from datetime import datetime
import dateutil.parser
import re

from currencies import MONEY_FORMATS
import pandas as pd
from schwifty import IBAN
import stdnum.lt

# numerical values

ALL_MONEY_FORMATS_SET = set(
    amount_format
    for currency in MONEY_FORMATS.values()
    for amount_format in currency.values()
)


def parses_as_amount(text):
    amount_pattern = re.compile(r"\d{1,3}(?:[',.]?\d{3})*(?:[,.]\d{2,8})")
    parsed_str = re.sub(amount_pattern, "{amount}", text)
    if parsed_str in ALL_MONEY_FORMATS_SET:
        return True
    return False


def parses_as_number(text):
    try:
        float(text)
        return True
    except ValueError:
        pass
    return parses_as_amount(text)


# standard numbers


def parses_as_IBAN(text):
    try:
        return IBAN(text)
    except ValueError:
        return False


def parses_as_VAT_registration(text):
    return stdnum.lt.vat.is_valid(text)


def parses_as_serial_number(text):
    return re.search(r"^[A-Za-z0-9/\-_\\ \t]+$", text)


def parses_as_invoice_number(text):
    return (
        parses_as_serial_number(text)
        and not parses_as_VAT_registration(text)
        and not parses_as_IBAN(text)
    )


# misc


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
    has_two_separators = re.search(r"\d+[^\d]\d+[^\d]\d+", text)
    return (min_default.date() == max_default.date()) and has_two_separators


PARSERS_DATAFRAME = pd.DataFrame(
    [
        (parses_as_full_date, 100, "date",),
        (parses_as_amount, 900, "amount"),
        (parses_as_number, 1000, "number"),
        (parses_as_IBAN, 200, "iban"),
        (parses_as_VAT_registration, 300, "vat_registration"),
        (parses_as_invoice_number, 400, "invoice_number"),
        (parses_as_serial_number, 500, "serial_number"),
    ],
    columns=["callable", "priority", "label"],
).sort_values("priority")
