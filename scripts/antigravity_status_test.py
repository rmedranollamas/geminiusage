#!/usr/bin/env python3
"""Tests for antigravity_status.py logic."""

import ssl
import unittest
from antigravity_status import _is_loopback, _get_ssl_context

class TestAntigravityStatus(unittest.TestCase):
    """Tests for the security fix and utility functions in antigravity_status.py."""

    def test_is_loopback(self) -> None:
        """Verifies that _is_loopback correctly identifies local addresses."""
        self.assertTrue(_is_loopback("https://127.0.0.1:1234/api"))
        self.assertTrue(_is_loopback("http://localhost:8080/"))
        self.assertTrue(_is_loopback("https://[::1]/test"))

        self.assertFalse(_is_loopback("https://google.com"))
        self.assertFalse(_is_loopback("http://192.168.1.1/"))
        self.assertFalse(_is_loopback("invalid-url"))
        self.assertFalse(_is_loopback(""))

    def test_get_ssl_context(self) -> None:
        """Verifies that _get_ssl_context returns unverified context only for local addresses."""
        local_url = "https://127.0.0.1:443/"
        remote_url = "https://example.com/"

        local_context = _get_ssl_context(local_url)
        remote_context = _get_ssl_context(remote_url)

        # check_hostname is False for unverified context
        self.assertFalse(local_context.check_hostname)
        self.assertEqual(local_context.verify_mode, ssl.CERT_NONE)

        # Default context should have verification enabled
        self.assertTrue(remote_context.check_hostname)
        self.assertEqual(remote_context.verify_mode, ssl.CERT_REQUIRED)

if __name__ == "__main__":
    unittest.main()
