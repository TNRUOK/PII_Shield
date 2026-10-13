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

def ssn_sanity(ssn: str, allow_test_ranges: bool = False) -> bool:
    """
    Return True if `ssn` (format NNN-NN-NNNN or 9 digits) passes basic SSN rules:
      - Area number (first 3 digits) ≠ 000, 666 (and < 900 unless allow_test_ranges=True)
      - Group number (middle 2 digits) ≠ 00
      - Serial number (last 4 digits) ≠ 0000
    """
    s = ssn.strip()
    m = re.fullmatch(r"(\d{3})-?(\d{2})-?(\d{4})", s)
    if not m:
        return False
    area, group, serial = m.group(1), m.group(2), m.group(3)
    area_n = int(area)
    if area_n == 0 or area_n == 666:
        return False
    if not allow_test_ranges and area_n >= 900:
        return False
    if group == "00":
        return False
    if serial == "0000":
        return False
    return True


# ---------------------------------------------------------------------------
# Polish PESEL validation
# ---------------------------------------------------------------------------

def pesel_check(pesel: str) -> bool:
    """
    Validate Polish PESEL number (11 digits with checksum) or partially masked
    form (e.g. '890412xxxxx').
    """
    clean = pesel.strip()
    if re.fullmatch(r"\d{6}[xX*]{5}", clean):
        # Partially masked format: first 6 digits must be valid YYMMDD
        m = int(clean[2:4])
        d = int(clean[4:6])
        return (1 <= (m % 20) <= 12) and (1 <= d <= 31)

    if not re.fullmatch(r"\d{11}", clean):
        return False

    weights = [1, 3, 7, 9, 1, 3, 7, 9, 1, 3]
    checksum = sum(int(clean[i]) * weights[i] for i in range(10)) % 10
    control = (10 - checksum) % 10
    return control == int(clean[10])


# ---------------------------------------------------------------------------
# Phone validation helper (offline phonenumbers wrapper)
# ---------------------------------------------------------------------------

def phone_validate(candidate: str) -> bool:
    """
    Optional phone validator using phonenumbers.
    Returns True if phonenumbers parses it as a possible or valid number.
    Does not reject test numbers (like 555-0xxx) if the regex matched.
    """
    clean = re.sub(r"[\s\(\)\.\-]", "", candidate)
    if len(clean) < 7 or len(clean) > 16:
        return False
    try:
        import phonenumbers
        # Try international parse if starts with '+'
        if candidate.strip().startswith("+"):
            parsed = phonenumbers.parse(candidate, None)
            return phonenumbers.is_possible_number(parsed)
        # Try US fallback
        parsed = phonenumbers.parse(candidate, "US")
        return phonenumbers.is_possible_number(parsed)
    except Exception:
        # Fallback: if phonenumbers fails to parse, return True if digit count is plausible
        return 7 <= len(clean) <= 15


# ---------------------------------------------------------------------------
# IPv4 octet range check
# ---------------------------------------------------------------------------

def ipv4_check(ip: str) -> bool:
    """
    Return True if `ip` is a valid IPv4 address (all octets 0-255).
    Leading zeros in octets are accepted (e.g. '192.168.01.01') to handle
    OCR noise and manual formatting differences.
    Also handles OCR separator variations (spaces, missing dot in fused octet).
    """
    ip = ip.strip().strip("<>")
    parts = [p for p in re.split(r"[\.\s,;]+", ip) if p]
    if len(parts) == 3 and len(parts[0]) == 4 and parts[0].isdigit():
        parts = [parts[0][:2], parts[0][2:]] + parts[1:]
    if len(parts) != 4:
        return False
    try:
        return all(0 <= int(p) <= 255 for p in parts if p)
    except ValueError:
        return False
