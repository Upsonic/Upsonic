"""
Module for handling caching of data using pickledb.
"""

import cloudpickle
import dill
import base64
import time
from typing import Optional, Tuple, Any
from .configuration import ClientConfiguration


def save_to_cache_with_expiry(data: Any, cache_key: str, expiry_seconds: int) -> None:
    """
    Save data to cache with expiration time.
    
    Args:
        data: Any data to store in cache
        cache_key: Unique identifier for the cached data
        expiry_seconds: Number of seconds until the cache expires
    """
    # Register module for better serialization
    the_module = dill.detect.getmodule(data)
    if the_module is not None:
        cloudpickle.register_pickle_by_value(the_module)
        
    expiry_time = int(time.time()) + expiry_seconds
    cache_data = {
        'data': data,
        'expiry_time': expiry_time
    }
    serialized_data = base64.b64encode(cloudpickle.dumps(cache_data)).decode('utf-8')
    ClientConfiguration.set(f"cache_{cache_key}", serialized_data)
    print(f"Saved data to cache with key {cache_key}, expires in {expiry_seconds} seconds")
    # test
    

def get_from_cache_with_expiry(cache_key: str) -> Optional[Any]:
    """
    Retrieve data from cache if not expired.
    
    Args:
        cache_key: Unique identifier for the cached data
        
    Returns:
        Cached data if found and not expired, None otherwise
    """
    serialized_data = ClientConfiguration.get(f"cache_{cache_key}")

    if serialized_data is None:
        print("No data found in cache for key:", cache_key)
        return None
    
    try:
        cache_data = cloudpickle.loads(base64.b64decode(serialized_data))
        current_time = int(time.time())
        
        if current_time > cache_data['expiry_time']:
            print(f"Cache expired for key: {cache_key}")
            # Clean up expired cache
            ClientConfiguration.set(f"cache_{cache_key}", None)
            return None
            
        print(f"Retrieved valid cached data for key: {cache_key}")
        return cache_data['data']
    except Exception as e:
        print(f"Error retrieving cached data: {str(e)}")
        return None