from typing import NewType

SessionId = NewType('SessionId', str)
"""
A distinct type representing the unique identifier for an InteractionSession.

While a 'str' at runtime, this alias allows static type checkers to
differentiate it from other string-based IDs, enabling type-safe
function overloading.
"""

UserId = NewType('UserId', str)
"""
A distinct type representing the unique identifier for a UserProfile.

While a 'str' at runtime, this alias allows static type checkers to
differentiate it from other string-based IDs, enabling type-safe
function overloading.
"""