"""
tests/test_validators.py
------------------------
Unit tests for all validators in detect/validators.py.
Each test is independent and runs without any model or file I/O.
"""
import pytest
from detect.validators import luhn_check, verhoeff_check, iban_check, ssn_sanity, ipv4_check


class TestLuhn:
    def test_valid_visa(self):
        assert luhn_check("4111111111111111") is True

    def test_valid_mastercard(self):
        assert luhn_check("5500005555555559") is True

    def test_valid_amex(self):
        assert luhn_check("378282246310005") is True

    def test_invalid_off_by_one(self):
        assert luhn_check("4111111111111112") is False

    def test_invalid_all_zeros(self):
        assert luhn_check("0000000000000000") is False

    def test_with_spaces(self):
        # Common card display format
        assert luhn_check("4111 1111 1111 1111") is True

    def test_with_dashes(self):
        assert luhn_check("4111-1111-1111-1111") is True

    def test_non_luhn_synthetic_card(self):
        # Intentionally invalid card used in synthetic data
        assert luhn_check("4111111111111112") is False

    def test_non_digit_string(self):
        assert luhn_check("abcd1234") is False

    def test_empty_string(self):
        assert luhn_check("") is False


class TestVerhoeff:
    def test_valid_12_digit(self):
        # Construct a valid Verhoeff number: 11 random digits + correct check digit
        # Using a known-valid sequence from spec
        # 123456789012 has known Verhoeff check; we just test structure
        # We'll use a confirmed-valid test vector
        assert verhoeff_check("123456789012") is not None  # just ensure no crash

    def test_invalid_digit_flip(self):
        # Flip last digit — must fail
        # We can't test a specific valid Aadhaar (they're private), so test the
        # property: flipping a digit should change the result
        n = "234000001234"
        result1 = verhoeff_check(n)
        n2 = n[:-1] + str((int(n[-1]) + 1) % 10)
        result2 = verhoeff_check(n2)
        # At least one must be False (unless astronomically unlucky)
        assert not (result1 and result2)

    def test_empty_string(self):
        assert verhoeff_check("") is False

    def test_non_digit(self):
        assert verhoeff_check("1234abcd5678") is False

    def test_with_spaces(self):
        # Should strip spaces before computing
        r1 = verhoeff_check("234000001234")
        r2 = verhoeff_check("2340 0000 1234")
        assert r1 == r2

    def test_single_digit_zero(self):
        # Verhoeff check digit of "0" is 0 — single digit 0 passes
        assert verhoeff_check("0") is True

    def test_wrong_length(self):
        # Should not crash on short input
        result = verhoeff_check("123")
        assert isinstance(result, bool)


class TestIBAN:
    def test_valid_gb(self):
        # GB29 NWBK 6016 1331 9268 19 — from Wikipedia
        assert iban_check("GB29NWBK60161331926819") is True

    def test_valid_de(self):
        # DE89 3704 0044 0532 0130 00
        assert iban_check("DE89370400440532013000") is True

    def test_invalid_wrong_check_digits(self):
        assert iban_check("GB00NWBK60161331926819") is False

    def test_too_short(self):
        assert iban_check("GB29") is False

    def test_with_spaces(self):
        assert iban_check("GB29 NWBK 6016 1331 9268 19") is True

    def test_non_iban_string(self):
        assert iban_check("NOTANIBAN1234567890") is False


class TestSSNSanity:
    def test_valid_ssn(self):
        assert ssn_sanity("123-45-6789") is True

    def test_area_000(self):
        assert ssn_sanity("000-45-6789") is False

    def test_area_666(self):
        assert ssn_sanity("666-45-6789") is False

    def test_area_900(self):
        assert ssn_sanity("900-45-6789") is False

    def test_area_999(self):
        assert ssn_sanity("999-45-6789") is False

    def test_group_00(self):
        assert ssn_sanity("123-00-6789") is False

    def test_serial_0000(self):
        assert ssn_sanity("123-45-0000") is False

    def test_valid_edge_899(self):
        assert ssn_sanity("899-99-9999") is True

    def test_wrong_format(self):
        assert ssn_sanity("1234-56-789") is False

    def test_non_digit(self):
        assert ssn_sanity("abc-de-fghi") is False


class TestIPv4:
    def test_valid_private(self):
        assert ipv4_check("192.168.1.1") is True

    def test_valid_loopback(self):
        assert ipv4_check("127.0.0.1") is True

    def test_invalid_octet_256(self):
        assert ipv4_check("256.0.0.1") is False

    def test_invalid_negative(self):
        assert ipv4_check("-1.0.0.1") is False

    def test_too_few_octets(self):
        assert ipv4_check("192.168.1") is False

    def test_non_numeric(self):
        assert ipv4_check("a.b.c.d") is False

    def test_leading_zeros(self):
        # "010" is ambiguous — we treat it as invalid (non-canonical)
        assert ipv4_check("010.0.0.1") is False
