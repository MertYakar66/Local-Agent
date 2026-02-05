"""Security hardening for the autonomous browser agent

This module provides:
- Input validation and sanitization
- URL safety checks (prevent SSRF, malicious redirects)
- Rate limiting to prevent abuse
- Security audit logging
- Emergency stop mechanisms
- Content security policies
"""

import hashlib
import hmac
import re
import secrets
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

from loguru import logger


class SecurityLevel(Enum):
    """Security levels for actions"""
    LOW = "low"           # Read-only, safe operations
    MEDIUM = "medium"     # Standard interactions (clicks, scrolls)
    HIGH = "high"         # Form submissions, data entry
    CRITICAL = "critical" # Financial, destructive, or sensitive


class ThreatType(Enum):
    """Types of security threats"""
    SSRF = "ssrf"                    # Server-side request forgery
    XSS = "xss"                      # Cross-site scripting in inputs
    INJECTION = "injection"          # Command/code injection
    MALICIOUS_URL = "malicious_url"  # Known malicious domains
    RATE_LIMIT = "rate_limit"        # Too many requests
    UNAUTHORIZED = "unauthorized"    # Unauthorized action attempt
    DATA_EXFIL = "data_exfiltration" # Potential data theft


@dataclass
class SecurityEvent:
    """A security event for audit logging"""
    timestamp: datetime
    event_type: str
    severity: SecurityLevel
    description: str
    source_ip: Optional[str] = None
    user_agent: Optional[str] = None
    url: Optional[str] = None
    action: Optional[str] = None
    blocked: bool = False
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "severity": self.severity.value,
            "description": self.description,
            "url": self.url,
            "action": self.action,
            "blocked": self.blocked,
            "details": self.details,
        }


class URLValidator:
    """Validates and sanitizes URLs to prevent security issues"""

    # Blocked URL schemes
    BLOCKED_SCHEMES = {
        "file", "ftp", "gopher", "data", "javascript",
        "vbscript", "about", "blob", "chrome", "chrome-extension",
    }

    # Private/internal IP ranges (SSRF prevention)
    PRIVATE_IP_PATTERNS = [
        r"^127\.",                          # Loopback
        r"^10\.",                           # Class A private
        r"^172\.(1[6-9]|2[0-9]|3[01])\.",   # Class B private
        r"^192\.168\.",                     # Class C private
        r"^169\.254\.",                     # Link-local
        r"^0\.",                            # Current network
        r"localhost",
        r"^::1$",                           # IPv6 loopback
        r"^fc00:",                          # IPv6 private
        r"^fe80:",                          # IPv6 link-local
    ]

    # Known malicious TLDs (high spam/malware rate)
    SUSPICIOUS_TLDS = {
        ".tk", ".ml", ".ga", ".cf", ".gq",  # Free TLDs often abused
        ".zip", ".mov",                      # Confusing TLDs
    }

    # Blocklist of known malicious domains (sample - expand as needed)
    BLOCKED_DOMAINS: Set[str] = set()

    @classmethod
    def validate_url(cls, url: str) -> Tuple[bool, Optional[str]]:
        """
        Validate a URL for security issues.

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not url:
            return False, "Empty URL"

        try:
            parsed = urlparse(url)

            # Check scheme
            scheme = parsed.scheme.lower()
            if scheme in cls.BLOCKED_SCHEMES:
                return False, f"Blocked URL scheme: {scheme}"

            if scheme not in ("http", "https"):
                return False, f"Invalid URL scheme: {scheme}"

            # Prefer HTTPS
            if scheme == "http":
                logger.warning(f"Insecure HTTP URL: {url}")

            # Check for SSRF (private IPs)
            hostname = parsed.hostname or ""
            for pattern in cls.PRIVATE_IP_PATTERNS:
                if re.match(pattern, hostname, re.IGNORECASE):
                    return False, f"SSRF attempt blocked: private IP {hostname}"

            # Check for suspicious TLDs
            for tld in cls.SUSPICIOUS_TLDS:
                if hostname.endswith(tld):
                    logger.warning(f"Suspicious TLD in URL: {url}")

            # Check blocklist
            if hostname in cls.BLOCKED_DOMAINS:
                return False, f"Blocked domain: {hostname}"

            # Check for IP address URLs (often suspicious)
            if re.match(r"^\d+\.\d+\.\d+\.\d+$", hostname):
                logger.warning(f"Direct IP URL (suspicious): {url}")

            return True, None

        except Exception as e:
            return False, f"URL parsing error: {str(e)}"

    @classmethod
    def sanitize_url(cls, url: str) -> str:
        """Sanitize a URL by removing dangerous components"""
        # Remove null bytes and control characters
        url = re.sub(r'[\x00-\x1f\x7f]', '', url)

        # Ensure proper scheme
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        return url.strip()


class InputSanitizer:
    """Sanitizes user inputs to prevent injection attacks"""

    # Characters that could be used for injection
    DANGEROUS_CHARS = {
        '<', '>', '"', "'", '\\', '\x00',
        '\n', '\r', '\t', '`', '$', '{', '}',
    }

    # Patterns that suggest injection attempts
    INJECTION_PATTERNS = [
        r"<script",
        r"javascript:",
        r"on\w+\s*=",           # Event handlers (onclick, onerror, etc.)
        r"eval\s*\(",
        r"document\.",
        r"window\.",
        r"\bexec\s*\(",
        r"\bsystem\s*\(",
        r";\s*rm\s+-",          # Shell commands
        r"\|\s*sh\b",
        r"&&\s*curl",
        r"wget\s+http",
    ]

    @classmethod
    def sanitize_text_input(cls, text: str, max_length: int = 10000) -> str:
        """Sanitize text input for form fields"""
        if not text:
            return ""

        # Truncate to max length
        text = text[:max_length]

        # Remove null bytes
        text = text.replace('\x00', '')

        # Check for injection patterns
        for pattern in cls.INJECTION_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                logger.warning(f"Potential injection blocked: {pattern}")
                # Remove the dangerous pattern
                text = re.sub(pattern, '[BLOCKED]', text, flags=re.IGNORECASE)

        return text

    @classmethod
    def is_safe_input(cls, text: str) -> Tuple[bool, Optional[str]]:
        """Check if input is safe (doesn't contain injection attempts)"""
        for pattern in cls.INJECTION_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return False, f"Injection pattern detected: {pattern}"

        return True, None


class RateLimiter:
    """Rate limiting to prevent abuse"""

    def __init__(
        self,
        max_requests: int = 100,
        window_seconds: int = 60,
        max_actions_per_minute: int = 30,
    ):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.max_actions_per_minute = max_actions_per_minute

        # Track requests per key (e.g., IP, user, task)
        self._requests: Dict[str, List[float]] = defaultdict(list)
        self._actions: Dict[str, List[float]] = defaultdict(list)

    def is_allowed(self, key: str = "default") -> Tuple[bool, Optional[str]]:
        """Check if a request is allowed under rate limits"""
        now = time.time()
        cutoff = now - self.window_seconds

        # Clean old entries
        self._requests[key] = [t for t in self._requests[key] if t > cutoff]

        # Check limit
        if len(self._requests[key]) >= self.max_requests:
            return False, f"Rate limit exceeded: {self.max_requests} requests per {self.window_seconds}s"

        # Record this request
        self._requests[key].append(now)
        return True, None

    def is_action_allowed(self, key: str = "default") -> Tuple[bool, Optional[str]]:
        """Check if a browser action is allowed"""
        now = time.time()
        cutoff = now - 60  # 1 minute window for actions

        self._actions[key] = [t for t in self._actions[key] if t > cutoff]

        if len(self._actions[key]) >= self.max_actions_per_minute:
            return False, f"Action rate limit: {self.max_actions_per_minute} actions per minute"

        self._actions[key].append(now)
        return True, None


class SecurityAuditLog:
    """Security-focused audit logging"""

    def __init__(self, log_dir: Optional[Path] = None):
        self.log_dir = log_dir or Path("./logs/security")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._events: List[SecurityEvent] = []
        self._current_log_file: Optional[Path] = None

    def log_event(self, event: SecurityEvent) -> None:
        """Log a security event"""
        self._events.append(event)

        # Also log to file
        log_file = self.log_dir / f"security_{datetime.now().strftime('%Y%m%d')}.log"

        with open(log_file, "a") as f:
            import json
            f.write(json.dumps(event.to_dict()) + "\n")

        # Log to standard logger with appropriate level
        if event.severity == SecurityLevel.CRITICAL:
            logger.error(f"[SECURITY] {event.description}")
        elif event.severity == SecurityLevel.HIGH:
            logger.warning(f"[SECURITY] {event.description}")
        else:
            logger.info(f"[SECURITY] {event.description}")

    def log_blocked_action(
        self,
        action: str,
        reason: str,
        url: Optional[str] = None,
        details: Optional[Dict] = None,
    ) -> None:
        """Log a blocked action"""
        event = SecurityEvent(
            timestamp=datetime.now(),
            event_type="blocked_action",
            severity=SecurityLevel.HIGH,
            description=f"Blocked action '{action}': {reason}",
            url=url,
            action=action,
            blocked=True,
            details=details or {},
        )
        self.log_event(event)

    def log_threat_detected(
        self,
        threat_type: ThreatType,
        description: str,
        url: Optional[str] = None,
    ) -> None:
        """Log a detected threat"""
        event = SecurityEvent(
            timestamp=datetime.now(),
            event_type=f"threat_{threat_type.value}",
            severity=SecurityLevel.CRITICAL,
            description=description,
            url=url,
            blocked=True,
        )
        self.log_event(event)

    def get_recent_events(self, hours: int = 24) -> List[SecurityEvent]:
        """Get security events from the last N hours"""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [e for e in self._events if e.timestamp > cutoff]

    def get_blocked_count(self, hours: int = 24) -> int:
        """Get count of blocked actions in last N hours"""
        return len([e for e in self.get_recent_events(hours) if e.blocked])


class EmergencyStop:
    """Emergency stop mechanism for the agent"""

    def __init__(self):
        self._stopped = False
        self._stop_reason: Optional[str] = None
        self._stop_time: Optional[datetime] = None
        self._blocked_urls: Set[str] = set()
        self._blocked_actions: Set[str] = set()

    def trigger_stop(self, reason: str) -> None:
        """Trigger emergency stop"""
        self._stopped = True
        self._stop_reason = reason
        self._stop_time = datetime.now()
        logger.critical(f"[EMERGENCY STOP] {reason}")

    def reset(self) -> None:
        """Reset emergency stop (requires explicit action)"""
        self._stopped = False
        self._stop_reason = None
        self._stop_time = None
        logger.info("[EMERGENCY STOP] Reset - agent can resume")

    def is_stopped(self) -> bool:
        """Check if emergency stop is active"""
        return self._stopped

    def get_status(self) -> Dict[str, Any]:
        """Get emergency stop status"""
        return {
            "stopped": self._stopped,
            "reason": self._stop_reason,
            "stop_time": self._stop_time.isoformat() if self._stop_time else None,
            "blocked_urls": list(self._blocked_urls),
            "blocked_actions": list(self._blocked_actions),
        }

    def block_url(self, url: str) -> None:
        """Permanently block a URL"""
        self._blocked_urls.add(urlparse(url).netloc)

    def block_action(self, action: str) -> None:
        """Block a specific action type"""
        self._blocked_actions.add(action)

    def is_url_blocked(self, url: str) -> bool:
        """Check if URL is blocked"""
        return urlparse(url).netloc in self._blocked_urls

    def is_action_blocked(self, action: str) -> bool:
        """Check if action type is blocked"""
        return action in self._blocked_actions


class ActionValidator:
    """Validates actions before execution"""

    # Actions that always require HITL approval
    ALWAYS_APPROVE = {
        "purchase", "buy", "checkout", "pay", "submit_payment",
        "delete_account", "close_account", "transfer_funds",
        "sign_contract", "agree_terms", "confirm_order",
    }

    # Maximum coordinate values (prevent out-of-bounds)
    MAX_COORDINATE = 10000

    # Dangerous URL patterns for form submissions
    DANGEROUS_FORM_PATTERNS = [
        r"/admin",
        r"/delete",
        r"/remove",
        r"/transfer",
        r"/payment",
        r"/checkout",
    ]

    @classmethod
    def validate_action(
        cls,
        action_type: str,
        target_element: Optional[Dict] = None,
        url: str = "",
        value: Optional[str] = None,
    ) -> Tuple[bool, Optional[str], SecurityLevel]:
        """
        Validate an action before execution.

        Returns:
            Tuple of (is_valid, error_message, security_level)
        """
        # Check emergency stop
        # (would need access to EmergencyStop instance)

        # Validate action type
        action_lower = action_type.lower()

        # Check if action requires approval
        if action_lower in cls.ALWAYS_APPROVE:
            return True, "Requires HITL approval", SecurityLevel.CRITICAL

        # Validate coordinates if present
        if target_element:
            bbox = target_element.get("bbox", [])
            if bbox:
                for coord in bbox:
                    if not isinstance(coord, (int, float)):
                        return False, f"Invalid coordinate type: {type(coord)}", SecurityLevel.HIGH
                    if coord < 0 or coord > cls.MAX_COORDINATE:
                        return False, f"Coordinate out of bounds: {coord}", SecurityLevel.HIGH

        # Validate fill value
        if action_lower == "fill" and value:
            is_safe, error = InputSanitizer.is_safe_input(value)
            if not is_safe:
                return False, error, SecurityLevel.HIGH

        # Check URL for dangerous patterns
        for pattern in cls.DANGEROUS_FORM_PATTERNS:
            if re.search(pattern, url, re.IGNORECASE):
                return True, f"Sensitive URL pattern: {pattern}", SecurityLevel.HIGH

        # Determine security level
        if action_lower in ("click", "fill", "submit"):
            level = SecurityLevel.MEDIUM
        elif action_lower in ("scroll", "hover", "wait"):
            level = SecurityLevel.LOW
        elif action_lower == "extract":
            level = SecurityLevel.LOW
        else:
            level = SecurityLevel.MEDIUM

        return True, None, level


class SecureSession:
    """Manages secure session tokens and authentication"""

    def __init__(self, secret_key: Optional[str] = None):
        self._secret_key = secret_key or secrets.token_hex(32)
        self._session_tokens: Dict[str, datetime] = {}
        self._token_lifetime = timedelta(hours=24)

    def generate_token(self, task_id: str) -> str:
        """Generate a secure session token for a task"""
        timestamp = datetime.now().isoformat()
        data = f"{task_id}:{timestamp}:{secrets.token_hex(16)}"

        # Create HMAC signature
        signature = hmac.new(
            self._secret_key.encode(),
            data.encode(),
            hashlib.sha256
        ).hexdigest()

        token = f"{data}:{signature}"
        self._session_tokens[token] = datetime.now()

        return token

    def validate_token(self, token: str) -> bool:
        """Validate a session token"""
        if token not in self._session_tokens:
            return False

        # Check expiration
        created = self._session_tokens[token]
        if datetime.now() - created > self._token_lifetime:
            del self._session_tokens[token]
            return False

        return True

    def revoke_token(self, token: str) -> None:
        """Revoke a session token"""
        self._session_tokens.pop(token, None)

    def cleanup_expired(self) -> int:
        """Remove expired tokens"""
        now = datetime.now()
        expired = [
            t for t, created in self._session_tokens.items()
            if now - created > self._token_lifetime
        ]
        for token in expired:
            del self._session_tokens[token]
        return len(expired)


# Global security instances
_url_validator = URLValidator()
_input_sanitizer = InputSanitizer()
_rate_limiter = RateLimiter()
_security_log = SecurityAuditLog()
_emergency_stop = EmergencyStop()
_action_validator = ActionValidator()


def get_security_log() -> SecurityAuditLog:
    """Get the global security audit log"""
    return _security_log


def get_emergency_stop() -> EmergencyStop:
    """Get the global emergency stop mechanism"""
    return _emergency_stop


def get_rate_limiter() -> RateLimiter:
    """Get the global rate limiter"""
    return _rate_limiter


def validate_url(url: str) -> Tuple[bool, Optional[str]]:
    """Validate a URL for security"""
    return _url_validator.validate_url(url)


def sanitize_url(url: str) -> str:
    """Sanitize a URL"""
    return _url_validator.sanitize_url(url)


def sanitize_input(text: str) -> str:
    """Sanitize user input"""
    return _input_sanitizer.sanitize_text_input(text)


def validate_action(
    action_type: str,
    target_element: Optional[Dict] = None,
    url: str = "",
    value: Optional[str] = None,
) -> Tuple[bool, Optional[str], SecurityLevel]:
    """Validate an action before execution"""
    return _action_validator.validate_action(action_type, target_element, url, value)
