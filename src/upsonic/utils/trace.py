import sentry_sdk as sentry_sdk_
from sentry_sdk.transport import Transport
from dotenv import load_dotenv

from upsonic.utils.package.get_version import get_library_version
from upsonic.utils.package.system_id import get_system_id


import os

load_dotenv()

the_dsn = os.getenv("UPSONIC_TELEMETRY", "https://7023ec3e0699da14a8013478e50b9142@o4508336623583232.ingest.us.sentry.io/4508607159599104")
the_environment = os.getenv("UPSONIC_ENVIRONMENT", "production")
the_release = f"upsonic@{get_library_version()}"
the_server_name = "upsonic_client"
the_sample_rate = float(os.getenv("UPSONIC_SENTRY_SAMPLE_RATE", "1.0"))

if the_dsn.lower() == "false":
    the_dsn = ""

sentry_sdk_.init(
    dsn=the_dsn,
    traces_sample_rate=the_sample_rate,
    release=the_release,
    server_name=the_server_name,
    environment=the_environment,
    enable_logs=True,
)

sentry_sdk_.set_user({"id": get_system_id()})




import logging

# Your existing logging setup
logger = logging.getLogger(__name__)

# These logs will be automatically sent to Sentry
logger.info('This will be sent to Sentry')

sentry_sdk = sentry_sdk_
print("Sentry initialized for Upsonic")