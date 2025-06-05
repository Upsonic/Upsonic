def agent_tool_register(upsonic_agent, agent, tasks):

    # If tasks is not a list
    if not isinstance(tasks, list):
        tasks = [tasks]

    for task in tasks:


        for tool in task.tools:
            agent.tool_plain(tool)

    return agent