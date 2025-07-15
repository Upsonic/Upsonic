def agent_tool_register(upsonic_agent, agent, tasks):

    # If tasks is not a list
    if not isinstance(tasks, list):
        tasks = [tasks]

    # Keep track of already registered tools to prevent duplicates
    if not hasattr(agent, '_registered_tools'):
        agent._registered_tools = set()

    for task in tasks:
        for tool in task.tools:
            # Create a unique identifier for the tool
            # Using id() to get unique object identifier
            tool_id = id(tool)
            
            # Only register if not already registered
            if tool_id not in agent._registered_tools:
                agent.tool_plain(tool)
                agent._registered_tools.add(tool_id)

    return agent