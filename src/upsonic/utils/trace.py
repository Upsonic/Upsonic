import sentry_sdk as sentry_sdk_
from sentry_sdk.transport import Transport
from dotenv import load_dotenv

from ..get_version import get_library_version
from ..system_id import get_system_id


import os

load_dotenv()

the_dsn = os.getenv("UPSONIC_TELEMETRY", "https://7023ec3e0699da14a8013478e50b9142@o4508336623583232.ingest.us.sentry.io/4508607159599104")
the_environment = os.getenv("UPSONIC_ENVIRONMENT", "production")

if the_dsn.lower() == "false":
    the_dsn = ""

sentry_sdk_.init(
    dsn=the_dsn,
    traces_sample_rate=1.0,
    release=f"upsonic@{get_library_version()}",
    server_name="upsonic_client",
    environment=the_environment,
)

sentry_sdk_.set_user({"id": get_system_id()})


sentry_sdk = sentry_sdk_