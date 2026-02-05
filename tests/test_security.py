"""Tests for the security module"""

import pytest
from datetime import datetime, timedelta

from src.utils.security import (
    URLValidator,
    InputSanitizer,
    RateLimiter,
    SecurityAuditLog,
    EmergencyStop,
    ActionValidator,
    SecureSession,
    SecurityLevel,
    ThreatType,
    validate_url,
    sanitize_url,
    sanitize_input,
    validate_action,
)


class TestURLValidator:
    """Test URL validation and sanitization"""

    def test_valid_https_url(self):
        """HTTPS URLs should be valid"""
        is_valid, error = URLValidator.validate_url("https://example.com/page")
        assert is_valid
        assert error is None

    def test_valid_http_url_with_warning(self):
        """HTTP URLs are valid but should log warning"""
        is_valid, error = URLValidator.validate_url("http://example.com/page")
        assert is_valid
        assert error is None

    def test_blocked_file_scheme(self):
        """File URLs should be blocked"""
        is_valid, error = URLValidator.validate_url("file:///etc/passwd")
        assert not is_valid
        assert "Blocked URL scheme" in error

    def test_blocked_javascript_scheme(self):
        """JavaScript URLs should be blocked"""
        is_valid, error = URLValidator.validate_url("javascript:alert(1)")
        assert not is_valid
        assert "Blocked URL scheme" in error

    def test_ssrf_localhost_blocked(self):
        """Localhost URLs should be blocked (SSRF)"""
        is_valid, error = URLValidator.validate_url("http://localhost/admin")
        assert not is_valid
        assert "SSRF" in error

    def test_ssrf_private_ip_blocked(self):
        """Private IP addresses should be blocked (SSRF)"""
        test_ips = [
            "http://127.0.0.1/",
            "http://10.0.0.1/",
            "http://172.16.0.1/",
            "http://192.168.1.1/",
        ]
        for url in test_ips:
            is_valid, error = URLValidator.validate_url(url)
            assert not is_valid, f"URL {url} should be blocked"
            assert "SSRF" in error

    def test_empty_url_invalid(self):
        """Empty URLs should be invalid"""
        is_valid, error = URLValidator.validate_url("")
        assert not is_valid
        assert "Empty URL" in error

    def test_sanitize_adds_https(self):
        """Sanitizer should add https:// to URLs without scheme"""
        result = URLValidator.sanitize_url("example.com")
        assert result == "https://example.com"

    def test_sanitize_removes_control_chars(self):
        """Sanitizer should remove control characters"""
        result = URLValidator.sanitize_url("https://example.com/\x00path")
        assert "\x00" not in result


class TestInputSanitizer:
    """Test input sanitization"""

    def test_normal_text_unchanged(self):
        """Normal text should pass through unchanged"""
        text = "Hello, this is a normal input"
        result = InputSanitizer.sanitize_text_input(text)
        assert result == text

    def test_script_tag_blocked(self):
        """Script tags should be blocked"""
        text = "<script>alert('xss')</script>"
        result = InputSanitizer.sanitize_text_input(text)
        assert "<script" not in result.lower()
        assert "[BLOCKED]" in result

    def test_javascript_url_blocked(self):
        """JavaScript URLs should be blocked"""
        text = "javascript:alert(1)"
        result = InputSanitizer.sanitize_text_input(text)
        assert "javascript:" not in result.lower()

    def test_event_handler_blocked(self):
        """Event handlers should be blocked"""
        text = 'onerror="alert(1)"'
        result = InputSanitizer.sanitize_text_input(text)
        assert "[BLOCKED]" in result

    def test_shell_command_blocked(self):
        """Shell commands should be blocked"""
        text = "; rm -rf /"
        result = InputSanitizer.sanitize_text_input(text)
        assert "[BLOCKED]" in result

    def test_null_bytes_removed(self):
        """Null bytes should be removed"""
        text = "hello\x00world"
        result = InputSanitizer.sanitize_text_input(text)
        assert "\x00" not in result
        assert result == "helloworld"

    def test_max_length_truncation(self):
        """Input should be truncated to max length"""
        text = "a" * 20000
        result = InputSanitizer.sanitize_text_input(text, max_length=100)
        assert len(result) == 100

    def test_is_safe_input_detects_injection(self):
        """is_safe_input should detect injection attempts"""
        is_safe, error = InputSanitizer.is_safe_input("<script>")
        assert not is_safe
        assert error is not None


class TestRateLimiter:
    """Test rate limiting"""

    def test_allows_requests_under_limit(self):
        """Requests under limit should be allowed"""
        limiter = RateLimiter(max_requests=10, window_seconds=60)
        for _ in range(10):
            allowed, _ = limiter.is_allowed("test")
            assert allowed

    def test_blocks_requests_over_limit(self):
        """Requests over limit should be blocked"""
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        for _ in range(5):
            limiter.is_allowed("test")

        allowed, error = limiter.is_allowed("test")
        assert not allowed
        assert "Rate limit exceeded" in error

    def test_different_keys_tracked_separately(self):
        """Different keys should have separate limits"""
        limiter = RateLimiter(max_requests=2, window_seconds=60)

        # Fill up key1
        for _ in range(2):
            limiter.is_allowed("key1")

        # key2 should still be allowed
        allowed, _ = limiter.is_allowed("key2")
        assert allowed

    def test_action_rate_limiting(self):
        """Action rate limiting should work"""
        limiter = RateLimiter(max_actions_per_minute=5)
        for _ in range(5):
            limiter.is_action_allowed("test")

        allowed, error = limiter.is_action_allowed("test")
        assert not allowed
        assert "Action rate limit" in error


class TestEmergencyStop:
    """Test emergency stop mechanism"""

    def test_initial_state_not_stopped(self):
        """Emergency stop should not be active initially"""
        stop = EmergencyStop()
        assert not stop.is_stopped()

    def test_trigger_stop(self):
        """Triggering stop should activate it"""
        stop = EmergencyStop()
        stop.trigger_stop("Test reason")
        assert stop.is_stopped()
        status = stop.get_status()
        assert status["stopped"]
        assert status["reason"] == "Test reason"

    def test_reset_stop(self):
        """Resetting stop should deactivate it"""
        stop = EmergencyStop()
        stop.trigger_stop("Test reason")
        stop.reset()
        assert not stop.is_stopped()

    def test_block_url(self):
        """Blocking a URL should work"""
        stop = EmergencyStop()
        stop.block_url("https://malicious.com/page")
        assert stop.is_url_blocked("https://malicious.com/other")

    def test_block_action(self):
        """Blocking an action should work"""
        stop = EmergencyStop()
        stop.block_action("delete")
        assert stop.is_action_blocked("delete")
        assert not stop.is_action_blocked("click")


class TestActionValidator:
    """Test action validation"""

    def test_valid_click_action(self):
        """Valid click action should pass"""
        is_valid, msg, level = ActionValidator.validate_action(
            action_type="click",
            target_element={"bbox": [100, 100, 200, 200], "description": "button"},
            url="https://example.com",
        )
        assert is_valid
        assert level == SecurityLevel.MEDIUM

    def test_critical_action_detected(self):
        """Purchase actions should be flagged as critical"""
        is_valid, msg, level = ActionValidator.validate_action(
            action_type="purchase",
            url="https://example.com/checkout",
        )
        assert is_valid  # Valid but requires approval
        assert level == SecurityLevel.CRITICAL
        assert "Requires HITL approval" in msg

    def test_invalid_coordinates_blocked(self):
        """Invalid coordinates should be blocked"""
        is_valid, msg, level = ActionValidator.validate_action(
            action_type="click",
            target_element={"bbox": [-100, 100, 200, 200]},
            url="https://example.com",
        )
        assert not is_valid
        assert "Coordinate out of bounds" in msg

    def test_fill_with_injection_blocked(self):
        """Fill with injection attempt should be blocked"""
        is_valid, msg, level = ActionValidator.validate_action(
            action_type="fill",
            target_element={"bbox": [100, 100, 200, 200]},
            url="https://example.com",
            value="<script>alert(1)</script>",
        )
        assert not is_valid
        assert "Injection pattern" in msg

    def test_sensitive_url_detected(self):
        """Sensitive URLs should be detected"""
        is_valid, msg, level = ActionValidator.validate_action(
            action_type="click",
            target_element={"bbox": [100, 100, 200, 200]},
            url="https://example.com/checkout/confirm",
        )
        assert is_valid
        assert level == SecurityLevel.HIGH


class TestSecureSession:
    """Test secure session management"""

    def test_generate_token(self):
        """Token generation should work"""
        session = SecureSession()
        token = session.generate_token("task_123")
        assert token is not None
        assert len(token) > 0

    def test_validate_valid_token(self):
        """Valid tokens should validate"""
        session = SecureSession()
        token = session.generate_token("task_123")
        assert session.validate_token(token)

    def test_validate_invalid_token(self):
        """Invalid tokens should not validate"""
        session = SecureSession()
        assert not session.validate_token("invalid_token")

    def test_revoke_token(self):
        """Revoked tokens should not validate"""
        session = SecureSession()
        token = session.generate_token("task_123")
        session.revoke_token(token)
        assert not session.validate_token(token)


class TestSecurityAuditLog:
    """Test security audit logging"""

    def test_log_event(self, tmp_path):
        """Events should be logged"""
        from src.utils.security import SecurityEvent

        log = SecurityAuditLog(log_dir=tmp_path / "security")
        event = SecurityEvent(
            timestamp=datetime.now(),
            event_type="test_event",
            severity=SecurityLevel.LOW,
            description="Test event",
        )
        log.log_event(event)

        # Check file was created
        log_files = list((tmp_path / "security").glob("*.log"))
        assert len(log_files) == 1

    def test_get_recent_events(self, tmp_path):
        """Recent events should be retrievable"""
        from src.utils.security import SecurityEvent

        log = SecurityAuditLog(log_dir=tmp_path / "security")
        event = SecurityEvent(
            timestamp=datetime.now(),
            event_type="test_event",
            severity=SecurityLevel.LOW,
            description="Test event",
        )
        log.log_event(event)

        recent = log.get_recent_events(hours=1)
        assert len(recent) == 1

    def test_blocked_count(self, tmp_path):
        """Blocked count should be accurate"""
        from src.utils.security import SecurityEvent

        log = SecurityAuditLog(log_dir=tmp_path / "security")

        # Log a blocked event
        event = SecurityEvent(
            timestamp=datetime.now(),
            event_type="blocked_action",
            severity=SecurityLevel.HIGH,
            description="Test blocked",
            blocked=True,
        )
        log.log_event(event)

        assert log.get_blocked_count(hours=1) == 1


class TestGlobalFunctions:
    """Test global convenience functions"""

    def test_validate_url_function(self):
        """Global validate_url function should work"""
        is_valid, _ = validate_url("https://example.com")
        assert is_valid

    def test_sanitize_url_function(self):
        """Global sanitize_url function should work"""
        result = sanitize_url("example.com")
        assert result.startswith("https://")

    def test_sanitize_input_function(self):
        """Global sanitize_input function should work"""
        result = sanitize_input("normal text")
        assert result == "normal text"

    def test_validate_action_function(self):
        """Global validate_action function should work"""
        is_valid, _, level = validate_action(
            action_type="click",
            url="https://example.com",
        )
        assert is_valid


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
