"""
detect/validators.py
--------------------
Standalone mathematical validators for specific PII types.
Each function returns True if the value passes the check.
These are pure functions with no side effects — easily unit-tested.
"""
from __future__ import annotations

import re
from typing import Optional


# ---------------------------------------------------------------------------
# Luhn algorithm (credit/debit card number validation)
# ---------------------------------------------------------------------------

def luhn_check(number: str) -> bool:
    """
    Return True if `number` (digits only, spaces/dashes stripped) satisfies
    the Luhn checksum (ISO/IEC 7812).

    Reference: https://en.wikipedia.org/wiki/Luhn_algorithm
    """
    digits = re.sub(r"[\s\-]", "", number)
    if not digits.isdigit() or len(digits) == 0:
        return False
    if set(digits) == {"0"}:
        return False
    total = 0
    reverse = digits[::-1]
    for i, d in enumerate(reverse):
        n = int(d)
        if i % 2 == 1:          # every second digit from the right (0-indexed → odd positions)
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


# ---------------------------------------------------------------------------
# Verhoeff algorithm (Aadhaar checksum)
# ---------------------------------------------------------------------------

# Multiplication table d[i][j] — element i multiplied by j in D5
_VERHOEFF_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]

# Permutation table p[i][j] — position permutation
_VERHOEFF_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]

# Inverse table inv[i]
_VERHOEFF_INV = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]


def verhoeff_check(number: str) -> bool:
    """
    Return True if `number` (digits only) satisfies the Verhoeff checksum.
    Used for Indian Aadhaar numbers (12 digits).

    Reference: https://en.wikipedia.org/wiki/Verhoeff_algorithm
    """
    digits = re.sub(r"[\s\-]", "", number)
    if not digits.isdigit():
        return False
    c = 0
    for i, d in enumerate(reversed(digits)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][int(d)]]
    return c == 0


# ---------------------------------------------------------------------------
# IBAN — ISO 13616 mod-97 check
# ---------------------------------------------------------------------------

def iban_check(iban: str) -> bool:
    """
    Return True if `iban` (spaces stripped, case-insensitive) passes the
    ISO 13616 mod-97 check.

    Reference: https://en.wikipedia.org/wiki/International_Bank_Account_Number
    """
    iban_clean = re.sub(r"\s", "", iban).upper()
    if len(iban_clean) < 15:
        return False
    # Move first 4 chars to end
    rearranged = iban_clean[4:] + iban_clean[:4]
    # Replace letters with digits: A=10, B=11, …, Z=35
    numeric = "".join(
        str(ord(ch) - ord("A") + 10) if ch.isalpha() else ch
        for ch in rearranged
    )
    try:
        return int(numeric) % 97 == 1
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# US SSN sanity check
# ---------------------------------------------------------------------------

def ssn_sanity(ssn: str) -> bool:
    """
    Return True if `ssn` (format NNN-NN-NNNN) passes basic SSN rules:
      - Area number (first 3 digits) ≠ 000, 666, or 900-999
      - Group number (middle 2 digits) ≠ 00
      - Serial number (last 4 digits) ≠ 0000

    Does NOT validate against the SSA's actual assignment records.
    """
    m = re.fullmatch(r"(\d{3})-(\d{2})-(\d{4})", ssn.strip())
    if not m:
        return False
    area, group, serial = m.group(1), m.group(2), m.group(3)
    area_n = int(area)
    if area_n == 0 or area_n == 666 or area_n >= 900:
        return False
    if group == "00":
        return False
    if serial == "0000":
        return False
    return True


# ---------------------------------------------------------------------------
# IPv4 octet range check
# ---------------------------------------------------------------------------

def ipv4_check(ip: str) -> bool:
    """
    Return True if `ip` is a valid IPv4 address (all octets 0-255).
    Leading zeros in octets are accepted (e.g. '192.168.01.01') to handle
    OCR noise and manual formatting differences.
    Note: The rule uses soft_penalty so private ranges still generate hits
    but at reduced confidence.
    """
    # Strip angle brackets or whitespace that might surround the IP
    ip = ip.strip().strip("<>")
    parts = ip.split(".")
    if len(parts) != 4:
        return False
    try:
        # int() strips leading zeros; check range only (don't require canonical form)
        return all(0 <= int(p) <= 255 for p in parts if p)
    except ValueError:
        return False
