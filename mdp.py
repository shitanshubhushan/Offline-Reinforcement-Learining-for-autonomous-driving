##### Note: Markov Decision Process
# S: the set of states
# A: the set of actions
# P: the dynamic transition model
# R: the reward function
# r: discount factor


def get_other_agents_unique_id(unique_id, current_time, uniqueTracks):
    """
    get other agents and closest three agents
    """

    # if the unique_id is not in the current time frame, then return empty lists
    try:
        uniqueTracks[unique_id].motionState[current_time]
    except:
        return [], []

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


def check_collision(agent_id, interactive_agents, current_time, uniqueTracks):
    flag = False

    return flag


def get_mdp_tuple(ego_id, interactive_agents, current_time, uniqueTracks):
    """
    get mdp tuple only if
    (1) interactive agents is not empty
    (2) the ego id is a car 
    """

    ### 0. pre check
    # 0-1 if interactive_agents is empty, then ego_id is not in the current time frame and return None
    if not interactive_agents:
        return None
    # 0-2. ego_id is not a car, return None
    if uniqueTracks[ego_id].agent_type == -1:
        return None
    # 0-3. edge cases
    if current_time == 4000:
        return None

    ### 1. state
    # ego vehicle id + interactive ids + ego info.
    egoMotionState = uniqueTracks[ego_id].motionState
    s = [ego_id] + interactive_agents + list(egoMotionState[current_time].values())
    # add interactive vehicles info.

    while len(interactive_agents) < 3:
        interactive_agents.append(0)
    for agent_id in interactive_agents:
        if agent_id != 0:
            s += list(uniqueTracks[agent_id].motionState[current_time].values())
        else:
            s += [0, 0, 0, 0, 0, 0]
    
    ### 2. action -> velocity in the next timestep
    next_time = current_time + 100
    a = [egoMotionState[next_time]['vx'], egoMotionState[next_time]['vy']]

    ### 3. reward
    r = 0
    if check_collision(ego_id, interactive_agents, current_time, uniqueTracks):
        r -= 100

    ### 4. next state
    # get next interactive agents
    _, interactive_agents_next = get_other_agents_unique_id(ego_id, next_time, uniqueTracks)
    # repeat step 1. to get next state
    s_next = [ego_id] + interactive_agents_next + list(egoMotionState[next_time].values())
    for agent_id in interactive_agents_next:
        s_next += list(uniqueTracks[agent_id].motionState[next_time].values())

    ### the last episode
    # 1 the last time frame
    if next_time == 4000:
        the_last_episode = True
    else:
        the_last_episode = False
    # 2 the next time frame will be ego vehicle's last time frame
    _, next_next_frame = get_other_agents_unique_id(ego_id, next_time + 100, uniqueTracks)
    if not next_next_frame:
        the_last_episode = True
    
    return (s, a, r, s_next, the_last_episode)