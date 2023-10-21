# Define the Markov Decision Process (MDP) class
class MDP:
    # Initialize the MDP object with given parameters
    def __init__(self, grid_size, walls, terminal_states, reward, transition_probabilities, discount_rate, epsilon, states):
        self.grid_size = grid_size
        self.walls = walls
        self.terminal_states = terminal_states
        self.reward = reward
        self.transition_probabilities = transition_probabilities
        self.discount_rate = discount_rate
        self.epsilon = epsilon
        self.states = states

    # Return the string representation of the MDP object
    def __str__(self):
        return str(self.__dict__)

# This function reads and parses the input from a given file
def parse_input_file(filename):
    # Read the file, removing empty lines and comments
    with open(filename, 'r') as file:
        lines = [line.strip() for line in file if line.strip() and not line.startswith("#")]

    # Initialize a dictionary to store the parsed data
    data = {}
    for line in lines:
        key, value = line.split(":")
        data[key.strip()] = value.strip()

    # Parse the data from the file into appropriate data structures
    size = tuple(reversed(list(map(int, data['size'].split()))))
    walls = [tuple(reversed(list(map(int, item.split())))) for item in data['walls'].split(',')]
    terminal_states = {tuple(reversed(list(map(int, item.split()[:2])))): float(item.split()[2]) for item in data['terminal_states'].split(',')}
    reward = float(data['reward'])
    transition_probabilities = list(map(float, data['transition_probabilities'].split()))
    discount_rate = float(data['discount_rate'])
    epsilon = float(data['epsilon'])
    states = [(i, j) for j in range(1, size[1] + 1) for i in range(1, size[0] + 1) if (i, j) not in walls]

    # Return a new MDP object with the parsed data
    return MDP(size, walls, terminal_states, reward, transition_probabilities, discount_rate, epsilon, states)

# Define possible actions
ACTIONS = ['up', 'down', 'left', 'right']

# Get the next state based on the current state and action
def next_state(s, a):
    # Update the state based on the chosen action
    if a == 'up':
        return (s[0]+1, s[1])
    if a == 'down':
        return (s[0]-1, s[1])
    if a == 'left':
        return (s[0], s[1]-1)
    if a == 'right':
        return (s[0], s[1]+1)
    return s

# Compute the Q-value for a state-action pair
def q_value(mdp, s, a, U):
    # Initialize the transition probabilities based on the action
    next_states_prob = []
    if a == 'up':
        next_states_prob = [('up', 0.8), ('left', 0.1), ('right', 0.1)]
    if a == 'down':
        next_states_prob = [('down', 0.8), ('left', 0.1), ('right', 0.1)]
    if a == 'left':
        next_states_prob = [('left', 0.8), ('up', 0.1), ('down', 0.1)]
    if a == 'right':
        next_states_prob = [('right', 0.8), ('up', 0.1), ('down', 0.1)]

    # Compute the Q-value based on the possible next states
    q_value_result = 0
    for next_a, prob in next_states_prob:
        s_prime = next_state(s, next_a)
        # Check if the next state is valid
        if s_prime[0] < 1 or s_prime[0] > mdp.grid_size[0] or s_prime[1] < 1 or s_prime[1] > mdp.grid_size[1] or s_prime in mdp.walls:
            s_prime = s
        reward = mdp.reward
        # Update the reward if it's a terminal state
        if s_prime in mdp.terminal_states:
            reward = mdp.terminal_states[s_prime]
        q_value_result += prob * (reward + mdp.discount_rate * U.get(s_prime, 0))

    return q_value_result

# Print the grid with utility values
def print_grid(mdp, U):
    for row in reversed(range(1, mdp.grid_size[0]+1)):
        for col in range(1, mdp.grid_size[1]+1):
            if (row, col) in mdp.walls:
                print("--------------", end=" ")
            else:
                print("{:.5f}".format(U.get((row, col), 0)), end=" ")
        print("\n")

# Perform the value iteration algorithm
def value_iteration(mdp):
    # Initialize the utility values
    U = {s: 0 for s in mdp.states}
    U_prime = U.copy()
    iteration = 0
    
    while True:
        U = U_prime.copy()
        delta = 0
        print(f"iteration: {iteration}")
        print_grid(mdp, U)
        iteration += 1
        for s in mdp.states:
            # Skip terminal states
            if s in mdp.terminal_states:
                continue
            U_prime[s] = max(q_value(mdp, s, a, U) for a in ACTIONS)
            delta = max(delta, abs(U_prime[s] - U[s]))

        # Check for convergence
        if delta <= mdp.epsilon * (1 - mdp.discount_rate) / mdp.discount_rate:
            break

    print("Final Values After Convergence:")
    print_grid(mdp, U_prime)

    return U_prime

# Compute the optimal policy using the utility values
def compute_policy(mdp, U):
    policy = {}
    for s in U.keys():
        # If it's a terminal state, mark it
        if s in mdp.terminal_states:
            policy[s] = "T"
        else:
            # Choose the action with the highest Q-value
            action = max(ACTIONS, key=lambda a: q_value(mdp, s, a, U))
            if action == "up":
                policy[s] = "N"
            elif action == "down":
                policy[s] = "S"
            elif action == "left":
                policy[s] = "W"
            elif action == "right":
                policy[s] = "E"
    return policy

# Main function to run the code
def main(filename):
    mdp = parse_input_file(filename)
    U = value_iteration(mdp)
    policy = compute_policy(mdp, U)

    print("Final Policy:")
    for row in reversed(range(1, mdp.grid_size[0]+1)):
        for col in range(1, mdp.grid_size[1]+1):
            state = (row, col)
            if state in mdp.walls:
                print("W", end=" ")
            else:
                print(policy.get(state, " "), end=" ")
        print("\n")




if __name__ == '__main__':
    main('mdp_input.txt')#Change file to input you want
