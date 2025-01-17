import cloudpickle
cloudpickle.DEFAULT_PROTOCOL = 2
import dill
import base64
import httpx
from typing import Any, List, Dict, Optional, Type, Union
from pydantic import BaseModel
import os
import tempfile

from io import BytesIO


class Others:
    def screenshot(self, show: bool = True, save_path: Optional[str] = None) -> Optional[bytes]:
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        """
        Take a screenshot using the server and optionally display it or save it.

        Args:
            show: Whether to display the screenshot using matplotlib
            save_path: Optional path to save the screenshot

        Returns:
            The screenshot bytes if save_path is not provided
        """
        # Get the screenshot from the server
        response = self.send_request("/others/take_screenshot", {}, method="GET", return_raw=True)
        
        if save_path:
            # Save the screenshot to the specified path
            with open(save_path, 'wb') as f:
                f.write(response)
        
        if show:
            # Display the screenshot using matplotlib
            img = mpimg.imread(BytesIO(response))
            plt.figure(figsize=(15, 10))
            plt.axis('off')
            plt.imshow(img)
            plt.show()
        
        if not save_path:
            return response
        
        return None
