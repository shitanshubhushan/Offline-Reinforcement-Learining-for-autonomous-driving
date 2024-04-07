##### Note: Markov Decision Process
# S: the set of states
# A: the set of actions
# P: the dynamic transition model
# R: the reward function
# r: discount factor


def get_other_agents_unique_id(unique_id, current_time, uniqueTracks):
    other_agents_id_list = []
    x  = uniqueTracks[unique_id].motionState[current_time]['x']
    y  = uniqueTracks[unique_id].motionState[current_time]['y']
    distance = {}

    case_id = unique_id // 100
    for agent_id in uniqueTracks:
        if agent_id // 100 == case_id and agent_id != unique_id:
            # if uniqueTracks[agent_id].agent_type == "car":
            value = uniqueTracks[agent_id].motionState.get(current_time, "Not Found")
            if value != "Not Found":
                other_agents_id_list.append(agent_id)
                x_prim = uniqueTracks[agent_id].motionState[current_time]['x']
                y_prim = uniqueTracks[agent_id].motionState[current_time]['y']
                distance[agent_id] = (y_prim - y) ** 2 + (x_prim - x) ** 2
    sorted_three_agetns = [item[0] for item in sorted(distance.items(), key=lambda item: item[1])][:3]
    return other_agents_id_list, sorted_three_agetns


def MDP_tuple(ego_id, interactive_agents, current_time, uniqueTracks):

    ### 1. state
    # ego vehicle id + interactive ids + ego info.
    egoMotionState = uniqueTracks[ego_id].motionState
    s = [ego_id] + interactive_agents + list(egoMotionState[current_time].values())
    # add interactive vehicles info.
    for agent_id in interactive_agents:
        s += list(uniqueTracks[agent_id].motionState[current_time].values())
    
    ### 2. action -> velocity in the next timestep
    next_time = current_time + 100
    a = [egoMotionState[next_time]['vx'], egoMotionState[next_time]['vy']]

    ### 3. reward
    r = 0

    ### 4. next state
    # get next interactive agents
    _, interactive_agents_next = get_other_agents_unique_id(ego_id, next_time, uniqueTracks)
    # repeat step 1. to get next state
    s_next = [ego_id] + interactive_agents_next + list(egoMotionState[next_time].values())
    for agent_id in interactive_agents_next:
        s_next += list(uniqueTracks[agent_id].motionState[next_time].values())

    ### the last time frame
    if next_time == 4000:
        the_last_episode = True
    else:
        the_last_episode = False

    return (s, a, r, s_next, the_last_episode)