# whisker_array_driver.py
# drives whisker array simulations and interacts with a control agent (e.g. RL agent)
#
import subprocess
import select
import socket
import struct
import math
import time
from typing import NamedTuple, Tuple
import traceback
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from time import perf_counter

class WhiskerArraySimEnv(gym.Env):
    """
    Gym environment for whisker array simulation with RL control.
    """

    HOST = '127.0.0.1'
    PORT = 5000

    def __init__(self, episode_time, 
                       rl_interval,
                       finish_line,*,
                       wake_width:float = 100,
                       chase_delay:int = 1000,
                       vel_max = 0.4, # max velocity of array m/s, or mm/ms
                       signal_limit = 4000, # expected limit of signal (lift/drag moment) from simulations, used to set observation space bounds
                       xloc_limit = 2000,  # expected limit of x location of the array
                       yloc_limit = 1000,  # expected limit of y location of the array
                       timeout_get:int = 10,
                       array_input = 'array_input.dat',
                       wageng = './wageng',
                       ccx = './ccx',
                       sim_steptime = 12.5, # milliseconds, simulation run at a fixed sampling rate of 80 Hz
                       ):
        """
        The initialization of the environment only sets the parameters, the actual simulations start in reset()

        episode_time: time for 1 tracking pass, milliseconds
        finish_line: episode ends if array passes this line, y location, mm
        rl_interval: frequency of RL communication, multiplier of simulation steps, equivalent reaction time is rl_interval/sample_rate

        wake_width: effective sensing range from path, mm
        chase_delay: time delay after object start before array starts moving, milliseconds
        timeout_get: timeout for getting env state, seconds

        """

        super().__init__()

        self.episode_time = episode_time
        self.finish_line = finish_line
        self.rl_interval = rl_interval
        self.wake_width  = wake_width
        self.chase_delay = chase_delay
        self.signal_limit = signal_limit
        self.xloc_limit = xloc_limit
        self.yloc_limit = yloc_limit
        self.timeout_get = timeout_get

        self.array_input = array_input
        self.wageng = wageng
        self.ccx = ccx
        self.sim_steptime = sim_steptime

        self.vel_max = vel_max
        self.vel_avg = 0.5 * vel_max # average velocity

        # todo check parameters validity

        # generate array mesh files and ccx inputs for whisker array, reused during reset
        self.num_whiskers = generate_whisker_array(self.array_input)
        copy_ccx_inputs(self.num_whiskers) 
        set_chase_delay(self.chase_delay) # this modifies input_read_flow.dat to set the starting flow frame

        self.path_data = calc_path_data(read_object_path_xy())

        # observation space: time, xloc, yloc, xvel, yvel + 2* num_whiskers (lift and drag moments)
        n_obs = 5 + 2*self.num_whiskers
        var_low = np.array([         0.0,       0.0,       0.0,-self.vel_max, -self.vel_max] + [-signal_limit]*2*self.num_whiskers)
        var_high= np.array([episode_time,xloc_limit,yloc_limit, self.vel_max,  self.vel_max] + [ signal_limit]*2*self.num_whiskers)

        obs_low  = np.tile(var_low, (self.rl_interval,1)) # repeat for rl_interval times, one observation includes rl_interval sim steps
        obs_high = np.tile(var_high,(self.rl_interval,1))
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float64)

        # action space: x-velocity, y-velocity
        act_low  = np.array([-self.vel_max, -self.vel_max])
        act_high = np.array([ self.vel_max,  self.vel_max])
        self.action_space = spaces.Box(low=act_low, high=act_high, dtype=np.float64)

        # observation buffer
        nt_buf = self.rl_interval+1 # num of sim steps in obs buffer
                                    # needs to be rl_interval+1 to store the state before the action
                                    # which is needed to compute reward for the last action
        self.state_buffer = np.zeros((nt_buf,n_obs))
        self.workers = []
        self.sock = None

        self.n_simstep_max = int(math.floor(self.episode_time/self.sim_steptime))
        self.n_simstep = 0 # number of simulation steps taken
        self.n_interact = 0 # number of interactions with the agent

        self.L0 = self.vel_avg * self.rl_interval * self.sim_steptime
        return

    def reset(self, seed=None, options=None):
        """
        reset the environment and start new simulations
        """
        tic = perf_counter()
        super().reset(seed=seed)
        self.close() # close previous simulations if any

        if not seed is None: # set random start y position
            pass

        # start new simulations
        self.sock = start_tcp_server(self.HOST,self.PORT)
        self.workers = start_whisker_simulations(self.num_whiskers, self.sock)
        self.sock.close()

        # clear state buffer
        self.state_buffer[:,:] = 0.0
        self.n_simstep = 1
        self.n_interact = 0

        end, _ = self.recv_append_sim_sample() # get initial state
        if end:
            raise RuntimeError('Failed to get initial state from simulations.')

        toc = perf_counter()
        print(f'Environment reset done in {toc - tic:.2f} seconds.\n')
        return self.state_buffer, {}
    
    def step(self, action):
        """
        Take one (RL) step in the environment. This includes rl_interval simulation steps.

        action: array-like, [x-velocity, y-velocity]

        The returned values conform to Gymnasium API:
        - observation (object): an environment-specific object representing your observation of the environment. here it is a 2D array of shape (nt_buf, nv_buf)
        - reward (float) : amount of reward returned after previous action
        - end (bool): whether the episode has ended, in which case further step() calls will return undefined results
        - truncated (bool): whether the episode was truncated (due to a time limit)
        - info (dict): contains auxiliary diagnostic information (helpful for debugging)
        """

        self.n_interact += 1
        cmd = f'MOV,{action[0]:.3f},{action[1]:.3f}'

        # run rl_interval simulation steps using the same action command
        for i in range(self.rl_interval):
            end, finish_line_reached = self.sim_step(cmd)
            if end:
                break
        
        if finish_line_reached: # compute reward for the last partial step
            num_steps = i+1
        else:
            num_steps = self.rl_interval

        if not end or finish_line_reached:
            reward = calc_step_reward(self.state_buffer, num_steps, self.path_data, self.wake_width, self.L0)
        else:
            reward = 0.0


        if self.n_simstep > self.n_simstep_max:
            truncated = True
        else:
            truncated = False

        observation = self.state_buffer[-self.rl_interval:,:]
        return observation, reward, end, truncated, {}

    def sim_step(self,cmd):
        """
        Take one simulation step in the environment. 
        """
        self.n_simstep+=1
        if (self.n_simstep>self.n_simstep_max):
            print('Episode time limit reached, ending ..')
            end = True
            return end
        
        # note that the simulation starts running with 0 moving speed (i.e. collecting signal at initial location)
        # it sends 1 state first then take 1 action command
        # for this stepping to work, the initial state should be collected first in reset()

        # send action command
        try:
            is_ok, _ = send_action_cmd(cmd,self.workers)
        except:
            traceback.print_exc()
            is_ok = False

        if not is_ok:
            print('Error sending action command to simulations.')
            end = True
            return end
        
        # put one sample in the buffer
        end, finish = self.recv_append_sim_sample()

        return end, finish
    
    def recv_append_sim_sample(self) -> tuple[bool, bool]:
        "receive one sample from simulation and put it in the buffer"

        finish = False
        try:
            state, is_ok = recv_whisker_array_state(self.workers,timeout=self.timeout_get)
        except:
            traceback.print_exc()
            is_ok = False

        if not is_ok:
            print('Error receiving state from simulations.')
            end = True
            return end, finish

        end = False
        # oldest state discarded, newest state placed at end
        self.state_buffer = np.roll(self.state_buffer,-1,axis=0) 
        state_vars = np.array([state.time, state.xloc, state.yloc, state.xvel, state.yvel])
        self.state_buffer[-1,:] = np.concatenate((state_vars, state.signal))
        print(f'  frame {self.n_simstep:<4d}: {state.time:12.1f}, {state.xloc:12.1f}, {state.yloc:12.1f}, {state.xvel:7.3f},{state.yvel:7.3f}')

        if state.xloc>self.finish_line:
            print('  Array passed finish line, ending ..')
            end = True
            finish = True
            
        return end, finish

    def close(self):
        for w in self.workers:
            w.socket.close()
            w.proc.terminate()
            try:
                w.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                w.proc.kill()

        return

# each worker corresponds to one whisker in the array
# Worker = namedtuple("Worker", ["proc", "socket", "addr"])
class Worker(NamedTuple):
    proc: subprocess.Popen
    socket: socket.socket
    addr: Tuple[str,int]

class WhiskerArrayState(NamedTuple):
    time: float
    xloc: float
    yloc: float
    xvel: float
    yvel: float
    signal: np.ndarray


def run_trail_tracking(workers:list[Worker], path_data, episode_time, finish_line, *,
                       wake_width:float = 100,
                       rl_interval:int = 5,
                       memory_time:int = 1000,
                       timeout_get:int = 30,
                       sim_steptime = 12.5,
                       vel_avg = 0.2):
    """
    Send move commands and get sensor data

    workers: whisker simulation workers
    path_data: object path data, n*3, [x,y,s]
    episode_time: time for 1 tracking pass, milliseconds
    finish_line: episode ends if array passes this line

    wake_width: effetive sensing range from path 
    rl_interval: frequency of RL communication, multiplier of simulation steps, equivalent reaction time is rl_interval/sample_rate
    memory_time: time length of memory of previous states
    timeout_get: timeout for getting env state, seconds
    """
    
    max_simtime = 10000
    if(episode_time > max_simtime):
        print(f'episode time reduced to {max_simtime}')
        print('To use longer episode time, modify FSI_case_001.inp first.')
        episode_time = max_simtime

    n_simstep_max = int(math.floor(episode_time/sim_steptime))

    print(f'Start tracking simulation of {n_simstep_max} steps.')

    state_var_names = ['time', 'xloc', 'yloc', 'xvel', 'yvel'] # 

    L0 = vel_avg * rl_interval * sim_steptime

    nt_buf = int(memory_time/sim_steptime) # num of sim steps in state buffer
    if(nt_buf<rl_interval+1): # the buffer length needs to be long enough to store the state before the previous command
        nt_buf = rl_interval+1
        print(f'Memory length too short for set {rl_interval}, state buffer length increased to {nt_buf}')
    
    num_whiskers = len(workers)
    nv_buf = len(state_var_names) + num_whiskers*2 # num of variables in state buffer
    state_buffer = np.zeros((nt_buf,nv_buf))

    cmd = 'MOV,0,0' #
    end = False
    n_simstep = 0
    n_interact = 0
    while(not end):
        n_simstep+=1

        if (n_simstep>n_simstep_max):
            print('Episode time limit reached, ending ..')
            break

        # get state first, the simulation starts running with 0 moving speed (i.e. collecting signal at initial location)
        try:
            state, is_ok = recv_whisker_array_state(workers,timeout=timeout_get)
        except:
            traceback.print_exc()
            end = True

        if not is_ok:
            print('Array simulation error')
            break
        else: # put state in the buffer
            # oldest state discarded, newest state placed at end
            state_buffer = np.roll(state_buffer,-1,axis=0) 
            state_vars = np.array([state.time, state.xloc, state.yloc, state.xvel, state.yvel])
            state_buffer[-1,:] = np.concatenate((state_vars, state.signal))
            print(f'  State: {state.time:12.1f}, {state.xloc:12.1f}, {state.yloc:12.1f}, {state.xvel:7.3f},{state.yvel:7.3f}')
        
        if n_simstep%rl_interval == 0:
            if n_interact == 0: # first interaction
                reward = 0
            else:

                reward = calc_step_reward(state_buffer, rl_interval, path_data, wake_width, L0)

            cmd = get_rl_action_cmd(state_buffer,reward)
            n_interact += 1

        if state.yloc>finish_line:
            print('Array passed finish line, ending ..')
            break
        try:
            is_ok, _ = send_action_cmd(cmd,workers)
        except:
            traceback.print_exc()
            end = True
    
    # todo:
    # episode reward
    # gracefully end workers


def generate_whisker_array(array_input, wageng='./wageng'):
    """
    Calls wageng to generate the whisker meshes based on array_input.dat
    """
    inp = open(array_input).readlines()

    num_whiskers = int(inp[5])
    cmd = [wageng, '-i', array_input]
    results = subprocess.run(cmd, check=True, text=True, capture_output=True)

    if(results.returncode == 0):
        print(f'Generated mesh files for {num_whiskers}-whisker array.')
    else:
        raise RuntimeError('Failed to generate whisker array.')
    
    return num_whiskers

def copy_ccx_inputs(num_whiskers):
    """
    Generate CCX input files from template.
    """
    template = 'FSI_case_001.inp'
    inp = open(template).readlines()

    iinc = -1
    for i,line in enumerate(inp):
        if 'whisker_001' in line:
            iinc = i
            break

    if (iinc == -1):
        raise ValueError(f'Can not find whisker_001 in {template}')

    for i in range(2,num_whiskers+1):
        with open(f'FSI_case_{i:03d}.inp','w') as f:
            inp[iinc] = f'*include,input=whisker_{i:03d}.beam.inp'
            f.writelines(inp)
    print('Copied CCX inputs.')
    return

def set_starting_position(xpos,ypos):
    """
    Set the starting position of the array in input-whisker
    """
    pass

def start_whisker_simulations(num_whiskers,server_socket:socket.socket,ccx='./ccx'):
    """
    Start the simulation processes and establish connection
    """
    #todo: check if input files are complete
    
    #
    timeout = 10 # seconds
    workers = []
    print('Launching and connecting to simulation workers ..')
    for i in range(1,num_whiskers+1):
        f = open(f'FSI_case_{i:03d}.log','w')
        cmd = [ccx, f'FSI_case_{i:03d}']
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        # print(f'Waiting for worker {i} to connect ..')
        deadline = time.time() + timeout
        while time.time()<deadline:
            ret = p.poll()
            if ret is not None:
                raise RuntimeError(f"Worker {i} exited with status {ret}")

            ready, _, _ = select.select([server_socket], [], [], 0.5)
            if ready:
                conn, addr = server_socket.accept() # this blocks until client connects
                break
        
        if conn is None:
            raise RuntimeError(f'Failed to connect worker {i}')
        else:
            # print(f'Worker {i} connected through {conn}\n')
            workers.append(Worker(p,conn,addr))

    return workers

def read_object_path_xy():
    """
    Get the prescribed moving path of the object (trail generator).
    """
    # get flow path
    lines = open('input_read_flow.dat').readlines()
    flow_path = lines[1].strip()
    # read object path
    path_xy = np.fromfile(flow_path+'/path_xy.dat',
                          dtype=np.float64,
                          sep=' ') # default is sep = '' for binary data
    
    return path_xy.reshape((-1,2))

def calc_path_data(path_xy):
    """
    Calculate arc length along the path.
    """
    ds = np.linalg.norm(np.diff(path_xy, axis=0),axis=1)
    s = np.pad(np.cumsum(ds),(1,0))
    path_data = np.concatenate((path_xy,s[:,None]),axis=1)
    return path_data

def set_chase_delay(t_delay):
    """
    Start array moving t_delay (ms) after object start

    Done by modifying the input_read_flow.dat
    """
    lines = open('input_read_flow.dat').readlines()

    interval = int(lines[3].split(' ')[0])
    tdump = float(lines[4].split(' ')[0]) # time between flow frames
    start = int(t_delay/tdump)*interval

    lines[3] = f'{interval} {start} \t\t! dump step interval,  start step\n'
    
    open('input_read_flow.dat','w').writelines(lines)

def start_tcp_server(host,port):
    # Start a TCP server
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # allow reuse
    sock.bind((host,port))
    sock.listen()
    print(f"Server listening on {host}:{port}")
    return sock

def tcp_send_frame(sock, data: bytes):
    """Send length-prefixed data."""
    # Prefix with length (4-byte int, littel endian)
    length_prefix = struct.pack("<I", len(data))
    sock.sendall(length_prefix + data)

def tcp_rcev_frame(sock:socket.socket,timeout:float=30) -> bytes:
    """Receive one length-prefixed message."""
    sock.settimeout(timeout)

    try:
        # Read 4-byte length prefix
        prefix = b""
        while len(prefix) < 4:
            chunk = sock.recv(4 - len(prefix))
            if not chunk:
                raise ConnectionError("Socket closed while reading length prefix")
            prefix += chunk
        (msg_len,) = struct.unpack("<I", prefix) # < means little endian
        # print(f'Reading frame of {msg_len} bytes ..')

        # Now read exactly msg_len bytes
        data = b""
        while len(data) < msg_len:
            chunk = sock.recv(msg_len - len(data))
            if not chunk:
                raise ConnectionError("Socket closed while reading payload")
            data += chunk

        return data
    except socket.timeout:
        raise RuntimeError(f'tcp_rcev_frame timed out in {timeout:.1f} seconds.')

def get_rl_action_cmd(state_buffer, reward):
    # To be replaced by RL model
    print(f'Getting RL action .. reward {reward:.1f}')
    vel = 0.2
    angle = 0.1
    t = state_buffer[-1,0] # current time
    T = 3000
    angle = 2*math.pi*(t/T)
    cmd = f'MOV,{math.fabs(math.cos(angle))*vel}, {math.sin(angle)*vel}'
    return cmd

def dummy_action(state_buffer, reward):
    # To be replaced by RL model
    print(f'  making dummy action ..')
    vel = 0.2
    angle = 0.1
    t = state_buffer[-1,0] # current time
    T = 3000
    angle = 2*math.pi*(t/T)
    
    return [math.fabs(math.cos(angle))*vel, math.sin(angle)*vel]

def send_action_cmd(cmd:str,workers:list[Worker]) -> Tuple[bool,bool]:
    """
    raise_on_exit: whether to raise error if any worker exited

    Action command to the array, uniformly applied to all whiskers.
    The accepted command is "action,x-velocity,y-velocity",
    action can either be "MOV" or "END"
    e.g. "MOV,0.2,0.2".

    The returned status is useful for waiting all workers to finish (in non-interactive mode)
    """
    is_all_running = True   # all workers are running
    is_any_running = False  # has running workers

    for i, w in enumerate(workers):
        if w.proc.poll() is None:
            tcp_send_frame(w.socket,cmd.encode('ascii'))
            is_any_running = True
        else:
            print(f'Failed to send action.\nWhisker simulation {i+1} exited with code {w.proc.returncode}.')
            is_all_running = False

    return is_all_running, is_any_running

def recv_whisker_array_state(workers:list[Worker], timeout:float) -> Tuple[WhiskerArrayState,bool]:
    """Collect data from all workers via TCP.
    
    Protocol:
        Each worker sends a comma separted string, including
        time,xloc,yloc,xvel,yvel,lift_moment,drag_moment
    """
    state = None
    try:
        n = len(workers)
        signal = np.zeros((n*2,))
        is_simulation_ok = True
        
        for i, w in enumerate(workers):
            if w.proc.poll() is None:
                data = tcp_rcev_frame(w.socket,timeout).decode('ascii')
                data = np.fromstring(data,dtype=np.float64,sep=',')
                # print(data)
                signal[i*2:i*2+2] = data[-2:]
            else:
                print(f'\nFailed to receive status.\nWhisker simulation {i+1} exited with code {w.proc.returncode}.')
                is_simulation_ok = False
            
        state = WhiskerArrayState(
            time = data[0],
            xloc = data[1],
            yloc = data[2],
            xvel = data[3],
            yvel = data[4],
            signal = signal)
    except ConnectionError as e:
        # print(f'Connection error: {e}')
        is_simulation_ok = False
        raise RuntimeError('\nConnection to workers lost while receiving state. Probably simulation error, check FSI_case_00*.log files.\n')

    return state, is_simulation_ok

def calc_step_reward(state_buffer, num_steps, path_data, wake_width, L0):
    """
    Reward for every RL action

    path_data - array, n*3, [x,y,s] 
    wake_width - effective range to sense the wake
    """

    old_s = state_buffer[-num_steps-1,:]
    new_s = state_buffer[-1,:]

    d_old, i_old = distance_to_curve_2d(old_s[1], old_s[2], path_data[:,0:2])
    d_new, i_new = distance_to_curve_2d(new_s[1], new_s[2], path_data[:,0:2])
    l_old = path_data[i_old,2] # arc length along path
    l_new = path_data[i_new,2]

    if d_old < wake_width: # close enough to wake, should move closer to the path
        # When moved toward path
        # the further it was, the more reward
        # the closer it moved, the more reward

        # When moved away from path
        # the closer it was, the more penalty
        # the faraway it moved, the more penalty
        ka = 2.0
        kc = 0.5
        reward = ka*(d_old - d_new)/(d_old+kc*wake_width)*wake_width

        # progress reward (agent should move forward)
        # gauge progress against L0
        kl = 400.0 # todo: more vigorous way to calculate this
        reward += (l_new - l_old)/L0*kl

    else: # no sufficient info to guide action, neutral reward
        reward = 0
    
    print(f'  Step reward: time {state_buffer[-1,0]:.1f}, d_old {d_old:.1f}, d_new {d_new:.1f}, reward {reward:.1f}')
    return reward

def distance_to_curve_2d(x,y,path_xy):
    """
    Using distance to points as approximate (given the path is sampled densely enough)
    More robust way should be distance to segments
    """
    d = np.linalg.norm(path_xy - [x,y], axis=1)
    imin = np.argmin(d)

    return d[imin],imin

def test1():

    sock = start_tcp_server('127.0.0.1',5000)
    num_whiskers = generate_whisker_array('array_input.dat')

    copy_ccx_inputs(num_whiskers)
    set_chase_delay(1000) # milliseconds
    workers = start_whisker_simulations(num_whiskers, sock)
    sock.close()

    path_xy = read_object_path_xy()
    path_data = calc_path_data(path_xy)
    run_trail_tracking(workers, path_data, finish_line = 1000,episode_time=6000)
    
    return

def test_env():
    env = WhiskerArraySimEnv(episode_time=5000, finish_line=1100, rl_interval=4)

    num_episodes = 3
    for i in range(num_episodes):
        print(f'\nEpisode {i+1} ..')

        obs, info = env.reset()
        reward = 0
        done = False
        num_steps = 0
        while not done:
            num_steps += 1
            print(f'\n>>> RL Step {num_steps}')
            # action = env.action_space.sample()
            action = dummy_action(obs,reward)
            obs, reward, done, truncated, info = env.step(action)
            print(f'  Step summary: Action: [{action[0]:.3f}, {action[1]:.3f}], Reward: {reward}, Done: {done}, Truncated: {truncated}')

    env.close()
    return

if __name__ == "__main__":
    # test1()
    test_env()
