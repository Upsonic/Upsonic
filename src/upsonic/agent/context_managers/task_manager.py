from contextlib import asynccontextmanager
from typing import Dict, Any, Optional

import mimetypes
from upsonic.storage.session.llm import Artifact

class TaskManager:
    def __init__(self, task, agent, *, turn_data: Optional[Dict[str, Any]] = None):
        self.task = task
        self.agent = agent
        self.model_response = None
        self.turn_data = turn_data

        
    def process_response(self, model_response):
        self.model_response = model_response

        if self.turn_data is not None and self.model_response is None:
            if hasattr(self.task, 'error_message') and self.task.error_message:
                self.turn_data['error'] = str(self.task.error_message)

        return self.model_response

    async def _handle_attachments(self):
        """
        Checks for attachments in the task, persists them using the
        configured storage provider, and logs their metadata.
        """
        if not self.agent.storage or not self.task.attachments:
            return

        if not self.agent.conversation_id:
            print("[Artifacts] Warning: conversation_id not set on agent. Cannot log artifact metadata.")
            return

        print(f"[Storage] Found {len(self.task.attachments)} attachment(s) to process.")

        for attachment_path in self.task.attachments:
            try:
                artifact = Artifact(
                    conversation_id=self.agent.conversation_id,
                    storage_uri="",
                    mime_type=mimetypes.guess_type(attachment_path)[0] or "application/octet-stream",
                )
                
                with open(attachment_path, "rb") as f:
                    binary_data = f.read()
                
                storage_uri = self.agent.storage.store_artifact_data(
                    artifact_id=artifact.artifact_id,
                    conversation_id=artifact.conversation_id,
                    binary_data=binary_data
                )
                
                artifact.storage_uri = storage_uri
                
                self.agent.storage.log_artifact(artifact)
                
                print(f"[Storage] Successfully stored and logged attachment: {attachment_path}")

            except FileNotFoundError:
                print(f"[Storage] ERROR: Attachment file not found at '{attachment_path}'. Skipping.")
            except Exception as e:
                print(f"[Storage] ERROR: A failure occurred while processing attachment '{attachment_path}': {e}")

    @asynccontextmanager
    async def manage_task(self):
        # Start the task
        self.task.task_start(self.agent)

        await self._handle_attachments()
        
        try:
            yield self
        finally:
            # Set task response and end the task if we have a model response
            if self.model_response is not None:
                self.task.task_response(self.model_response)
                self.task.task_end() 