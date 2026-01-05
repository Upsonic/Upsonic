import asyncio
from typing import Optional
from pydantic import BaseModel, Field
from upsonic import Agent, Chat


class UserProfile(BaseModel):
    name: str = Field(description="The user's name as mentioned in the conversation")
    
    # Use explicit fields instead of dict - this works!
    theme_preference: Optional[str] = Field(
        default=None,
        description="User's UI theme preference (e.g., 'dark mode', 'light mode')"
    )
    programming_language: Optional[str] = Field(
        default=None,
        description="User's preferred programming language (e.g., 'Python', 'JavaScript')"
    )


async def main():
    agent = Agent("openai/gpt-4o")

    chat = Chat(
        session_id="session1",
        user_id="user1",
        agent=agent,
        user_profile_schema=UserProfile,
        user_analysis_memory=True
    )

    # Send messages that will trigger user analysis
    response1 = await chat.invoke("Hi, I'm Alice and I prefer dark mode and Python programming")
    print(f"Response 1: {response1}\n")

    response2 = await chat.invoke("What do you know about my preferences?")
    print(f"Response 2: {response2}\n")

    # Retrieve and display extracted profile data
    from upsonic.session.agent import AgentSession
    import json
    
    session = await chat._memory.storage.read_async(chat.session_id, AgentSession)
    if session and session.user_profile:
        print("=" * 60)
        print("Extracted Profile Data:")
        print("=" * 60)
        profile_data = session.user_profile
        print(json.dumps(profile_data, indent=2))
        
        # Reconstruct and display
        try:
            custom_profile = UserProfile.model_validate(profile_data)
            print("\n✅ Extracted Profile:")
            print(f"  Name: {custom_profile.name}")
            print(f"  Theme Preference: {custom_profile.theme_preference}")
            print(f"  Programming Language: {custom_profile.programming_language}")
        except Exception as e:
            print(f"\n❌ Error validating profile: {e}")
            print(f"Raw profile data: {profile_data}")

    print(f"\nTotal cost: ${chat.total_cost:.4f} | Tokens: {chat.input_tokens + chat.output_tokens:,}")


if __name__ == "__main__":
    asyncio.run(main())