from cmath import sqrt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def check_collision(curr_egoMotionState, interactive_egoMotionStates):
    """
    Check for collisions between the ego agent and interactive agents at the given time.
    input:
        ego_id: ego vehicle's unique id.
        interactive_agents: List of unique IDs of other agents to check for collision.
        current_time: The current time step to check for collisions.
        uniqueTracks: uniqueTrack objects with it's meta data
    
    returns:
        bool: True if a collision is detected, False otherwise.
    """
    # ego agent's state and features
    ego_x = curr_egoMotionState['x']
    ego_y = curr_egoMotionState['y']
    ego_psi = curr_egoMotionState['psi_rad']
    ego_length = curr_egoMotionState['length']
    ego_width = curr_egoMotionState['width']
    ego_half_length = ego_length / 2
    ego_half_width = ego_width / 2

    # Check for collision with each interactive agent
    for interactive_state in interactive_egoMotionStates:
        interactive_id = interactive_state['unique_id']
        if interactive_id != 0:
            interactive_x = interactive_state['x']
            interactive_y = interactive_state['y']
            interactive_psi = interactive_state['psi_rad']
            interactive_length = interactive_state['length']
            interactive_width = interactive_state['width']
            interactive_type = interactive_state['agent_type']
            
            if interactive_type == 1:  # Car
                return car_collision(ego_x, ego_y, ego_length, ego_width, ego_psi, interactive_x, interactive_y,
                                     interactive_length, interactive_width, interactive_psi)
            else:
                # Check point collision for pedestrians/bicycles
                # Assuming a nominal width and length for pedestrians/bicycles
                pedestrian_buffer = - 0.1  # buffer for pedestrians/bicycles
                if (abs(ego_x - interactive_x) < ego_half_width + pedestrian_buffer and
                    abs(ego_y - interactive_y) < ego_half_length + pedestrian_buffer):
                    print('non_car_collision: ', interactive_id)
                    return True

    return False  # No collision detected


def point_inside_rectangle(point, rectangle_min, rectangle_max):
    """
    Check if a point is inside a rectangle
    input:
        point: a point (x, y)
        rectangle_min: x_min, y_min
        rectangle_max: x_max, y_max
    output:
        nool: True if point in rectangle, False otherwise
    """
    return rectangle_min[0] <= point[0] <= rectangle_max[0] and rectangle_min[1] <= point[1] <= rectangle_max[1]

def point_in_rotated_rectangle(point, center, length, width, angle):
    """
    Check if a point is inside a rotated rectangle
    input:
        point: a point
        center: rectangle center
        length: rectangle length
        width: rectangle width
        angle: rectangle angle [rad]
    output:
        bool: True if point in rotated rectangle, False otherwise.
    """
    c, s = np.cos(angle), np.sin(angle)
    r = np.array([[c, -s], [s, c]])
    ru = r.dot(point - center)
    return point_inside_rectangle(ru, [-length/2, -width/2], [length/2, width/2])


def has_corner_inside(rectangle1, rectangle2):
    """
    Check if rect1 has a corner inside rect2
    input:
        rectangle1: (center, length, width, angle)
        rectangle2: (center, length, width, angle)
    output:
        bool: True if has cornel inside, False otherwise.
    """
    (c1, l1, w1, a1) = rectangle1
    (c2, l2, w2, a2) = rectangle2
    c1 = np.array(c1)
    l1v = np.array([l1/2, 0])
    w1v = np.array([0, w1/2])
    r1_points = np.array([[0, 0],
                          - l1v, l1v, -w1v, w1v,
                          - l1v - w1v, - l1v + w1v, + l1v - w1v, + l1v + w1v])
    c, s = np.cos(a1), np.sin(a1)
    r = np.array([[c, -s], [s, c]])
    rotated_r1_points = r.dot(r1_points.transpose()).transpose()
    return any([point_in_rotated_rectangle(c1+np.squeeze(p), c2, l2, w2, a2) for p in rotated_r1_points])


def rotated_rectangles_intersect(rectangle1, rectangle2):
    """
    check if two rotated rectangles intersect
    input:
        rectangle1: (center, length, width, angle)
        rectangle2: (center, length, width, angle)
    output:
        bool: True if intersect, False otherwise.
    """
    return has_corner_inside(rectangle1, rectangle2) or has_corner_inside(rectangle2, rectangle1)

def car_collision(ego_x, ego_y, ego_length, ego_width, ego_psi, interactive_x, interactive_y,
                  interactive_length, interactive_width, interactive_psi):
    if rotated_rectangles_intersect(((ego_x, ego_y), 0.95 * ego_length, 0.95 * ego_width, np.arccos(ego_psi)),
                                    ((interactive_x, interactive_y), 0.95 * interactive_length, 0.95 * interactive_width, np.arccos(interactive_psi))):
        return True
    return False


def get_reward(curr_egoMotionState, interactive_egoMotionStates, v_max):
    """
    reward function
    input:
        #ego_id
        interactive_agents
        current_time
        #uniqueTracks
        v_max: maximum velocity over all dataset

    output:
        total reward
    """

    # curr_egoMotionState = uniqueTracks[ego_id].motionState[current_time]
    # closest_egoMotionState = uniqueTracks[interactive_agents[0]].motionState[current_time]
    # velocity reward 
    vx = curr_egoMotionState['vx']
    vy = curr_egoMotionState['vy']
    v = (vx ** 2 + vy ** 2) ** 0.5
    sigma = 0.1
    rv_offset = -1
    rv_scale = 2
    rv = rv_scale/(sigma*sqrt(2*np.pi))*np.exp(-0.5*((v-v_max)/sigma)**2)+rv_scale*rv_offset
#     rv = 0.5 * v / v_max

    # position reward
    x = curr_egoMotionState['x']
    y = curr_egoMotionState['y']

    agent_closest = interactive_egoMotionStates[0]
    if agent_closest:
        x_closest = agent_closest['x']
        y_closest = agent_closest['y']

        distance_closest = np.sqrt((x_closest-x)**2 + (y_closest-y)**2) #b

        x_next = x + vx*0.1
        y_next = y + vy*0.1

        distance_next = np.sqrt((x_next-x)**2 + (y_next-y)**2) #a
        distance_next_closest = np.sqrt((x_next-x)**2 + (y_next-y)**2) #c

        alpha = np.arccos((distance_next**2 + distance_closest**2 - distance_next_closest**2)/(2*distance_next*distance_closest))

        angle_factor = 0
        if alpha>-30 and alpha<30:
            angle_factor = -1
            new_mean = np.min((0.5*distance_closest/0.1),v_max) #Mean of new reward distribution is minimum of v_max and half the velocity to close distance between ego and object
            rv = rv_scale/(sigma*sqrt(2*np.pi))*np.exp(-0.5*((v-new_mean)/sigma)**2)+rv_scale*rv_offset
        rp = distance_closest * angle_factor

        # rv = rv*distance_closest #If there is a closest object, the greater the distance, the greater the reward for high velocity
        # rv = rv_scale/(sigma*sqrt(2*np.pi))*np.exp(-0.5*((v-(v_mean+distance_closest/0.1))/sigma)**2)+rv_scale*rv_offset
    else:
        rp = 0

    # collision reward
    rc = -100 if check_collision(curr_egoMotionState, interactive_egoMotionStates) else 0

    return rv + rc + rp


def reward(r_p, r_v, r_C, p, p_hat, v, v_hat, C):
  '''Reward function for RL system
  INPUTS: 
          r_p: Hyperparameter, factor for position term in reward equation
          r_v: Hyperparameter, factor for velocity term in reward equation
          r_C: Hyperparameter, factor for collision term in reward equation
          p: Actual position of objects over next_steps
          p_hat: RL agent planned position of ego over next_steps
          v: Actual velocity of ego over next_steps
          v_hat: RL agent planned velocity of ego over next_steps
          C: Number of collisions over next_steps
          next_steps: Hyperparameter: Number of timesteps to consider in application of reward function
  OUTPUTS: 
          R: total reward'''
  
  # How to bound the R_p and R_v terms is something we should investigate in our training iterations
        # INstead of next_steps, need current state only - next steps goes away, norm goes away
#   p = p[0:next_steps-1]
#   p_hat = p_hat[0:next_steps-1]
#   v = v[0:next_steps-1]
#   v_hat = v_hat[0:next_steps-1]

  R_p = r_p/((p-p_hat)**2) #This needs to be bounded per the feedback on our progress report
  R_p = np.max(R_p, 100) #Bound to 100 (should we make this a hyperparameter?)

  R_v = r_v/((v-v_hat)**2) #This needs to be bounded per the feedback on our progress report
  R_v = np.max(R_v, 100) #Bound to 100 (should we make this a hyperparameter?)

  R_C = -r_C*np.sum(C)

  R = R_p + R_v + R_C
  
  return R