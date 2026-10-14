"""
Synchronous usage of ValkeyProvider.

The sync API is provided automatically by the base class — no extra
implementation needed. Useful for scripts and notebooks.

Requirements:
- Valkey 9.1+ with valkey-search module loaded
- pip install "upsonic[valkey]"
"""

from upsonic.vectordb import ValkeyProvider, ValkeyConfig
from upsonic.vectordb.config import ConnectionConfig, Mode, DistanceMetric


# Configure and connect (sync)
config = ValkeyConfig(
    vector_size=128,
    collection_name="sync_example",
    key_prefix="sync:",
    connection=ConnectionConfig(mode=Mode.LOCAL, host="localhost", port=6379),
    distance_metric=DistanceMetric.COSINE,
)

provider = ValkeyProvider(config)
provider.connect()
provider.create_collection()

# Upsert
provider.upsert(
    vectors=[[0.1] * 128, [0.9] * 128],
    ids=["a", "b"],
    chunks=["First document about AI", "Second document about databases"],
)

# Search
import time
time.sleep(0.3)  # Allow index to update

results = provider.search(
    query_vector=[0.1] * 128,
    top_k=2,
)
for r in results:
    print(f"[{r.score:.3f}] {r.id}: {r.text}")

# Cleanup
provider.delete_collection()
provider.disconnect()
