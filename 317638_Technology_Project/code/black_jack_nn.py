import random as rand
import numpy as np
import matplotlib.pyplot as plt
#-----------define neural network stuff--------


#----activation funcs----------

def relu(Z, alpha=0.01):
    return np.where(Z > 0, Z, alpha * Z)

def relu_derivative(Z, alpha=0.01):
    return np.where(Z > 0, 1.0, alpha)

def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z))  # subtract max for numerical stability
    return exp_Z / exp_Z.sum()

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))


def cross_entropy(predicted,target):
    predicted = np.clip(predicted,1e-8,1 - 1e-8)
    return -np.sum(target*np.log(predicted))

def cross_entropy_gradient(predicted,target):
    return predicted-target

def mse(predicted, target):
    return np.mean((predicted-target)**2)

def mse_gradient(predicted, target):
    return 2*(predicted-target)/len(predicted)
 

#----------- nn handling-----------------

def train_step(network, x, target):
    raw_output = network.forward(x)
    prediction = softmax(raw_output)
    loss = cross_entropy(prediction, target)
    grad = cross_entropy_gradient(prediction,target)
    network.backward(grad)
    return loss

def update(W, b, dL_dW, dL_db, lr=0.01):
    W -= lr * dL_dW
    b -= lr * dL_db
    return W, b

#-defined seperate layers for readability, but kinda awful
#-could be good flexibilty for future approaches?
class Layer:
    def __init__(self, n_in, n_out):
        self.W = np.random.randn(n_in, n_out) * np.sqrt(2 / n_in)
        self.b = np.zeros(n_out)
        self.X = None

    def forward(self, X, activation):
        self.X = X
        self.Z = np.dot(X, self.W) + self.b
        return activation(self.Z)

    def backward(self, dL_dout, activation_deriv, lr):
        dL_dZ = dL_dout * activation_deriv(self.Z)
        dL_dW = np.outer(self.X, dL_dZ)
        dL_dX = np.dot(self.W, dL_dZ)
        dL_db = dL_dZ
        self.W -= lr * dL_dW
        self.b -= lr * dL_db
        return dL_dX
    
    def save(self, path):
        data = {}
        for i, layer in enumerate(self.layers):
            data[f'W{i}'] = layer.W
            data[f'b{i}'] = layer.b
        np.savez(path, **data)

    def load(self, path):
        data = np.load(path + '.npz')
        for i, layer in enumerate(self.layers):
            layer.W = data[f'W{i}']
            layer.b = data[f'b{i}']

    
class Network:
    def __init__(self, layer_sizes):
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i+1]))

    def forward(self, X):
        out = X
        for i, layer in enumerate(self.layers):
            # relu on hidden layers, linear on last layer
            if i < len(self.layers) - 1:
                out = layer.forward(out, relu)
            else:
                out = layer.forward(out, lambda z: z)
        return out

    def backward(self, grad, lr=0.0001, clip=1.0):
        norms = []
        for i, layer in enumerate(reversed(self.layers)):
            grad = np.clip(grad, -clip, clip)  # clip before each layer
            grad = layer.backward(grad, relu_derivative, lr)
            norm = np.linalg.norm(grad)
            norms.append((i, norm))
        return norms

    def save(self, path):
        data = {}
        for i, layer in enumerate(self.layers):
            data[f'W{i}'] = layer.W
            data[f'b{i}'] = layer.b
        np.savez(path, **data)

    def load(self, path):
        data = np.load(path + '.npz')
        for i, layer in enumerate(self.layers):
            layer.W = data[f'W{i}']
            layer.b = data[f'b{i}']


class TrainingMonitor:
    def __init__(self):
        self.losses=[]
        self.grad_norms=[]
    
    def record(self, loss, grad_norms):
        self.losses.append(loss)
        self.grad_norms.append(grad_norms)
    
    def report(self, epoch):
        if not self.losses: return
        if not self.grad_norms: return

        recent = np.mean(self.losses[-100:])
        print('epoch:', epoch, 'average loss=', recent)
        last = self.grad_norms[-1]

        if last is None: return
        for layer, norm in last:
            if norm < 1e-6:
                status = 'Vanishing'
            elif norm > 1000:
                status = 'Exploding'
            else:
                status = 'Ok'
            print('Layer:', layer, 'norm=', norm, status)
    
    def plot(self):
        return self.losses



#------------player nn input vector-----------

def p_vector(p_hand,d_upcard,softhand,p_bust,d_bust,confidence,deviation):
    if p_hand == 'bust':
        return None

    return np.array(
        [p_hand/21, d_upcard/11,float(softhand),p_bust/100,d_bust/100,confidence,deviation/10
        ],dtype=np.float64
    )



#----------player input handling---------

moves=['hit','stand','double','split']


def validate_input(vec, name):
    print('Name:',name,'shape=',vec.shape,'min=',vec.min(),'max=',vec.max())

    if np.any(np.isnan(vec)):
        print('Warning nan in',name)
    if vec.max()>2.0 or vec.min() < -2.0:
        print('Warning: values outside expected range in',name)


def soft_hand(hand):
    has_ace= 'ace' in hand
    if not has_ace:
        return False
    if type(check_hand(hand))!= str:
        return check_hand(hand)+10<=21
    else:
        return False


def extract_training(training_signal, deck):
    examples = []
    saved = save_card_ocrs()  # save current global state

    for deck_index, signal in training_signal.items():
        state = signal.get('state')
        if state is None:
            continue

        replay_deck_to(deck, deck_index)  # set global to correct state

        p_total = state['state']['p_total']
        upcard = state['state']['dealer_upcard']
        soft = state['state']['soft_flag']
        p_bust = bust_prob(p_total)
        d_bust = bust_prob(state['state']['d_total'])
        dev = deck_deviation()
        conf = confidence()

        x = p_vector(p_total, upcard, soft, p_bust, d_bust, conf, dev)

        target = np.zeros(4)
        for hit_count, weight in signal['action_distribution'].items():
            target[min(hit_count, 3)] += weight
        if target.sum() > 0:
            target /= target.sum()

        examples.append((x, target, signal['ev']))

    restore_card_ocrs(saved)  # restore global state after extraction
    return examples


#--------define card handling------------
card_types = [2,3,4,5,6,7,8,9,10,'jack','queen','king','ace']

def create_deck():
    deck=[]
    for _ in range(0,len(card_types)*4):
        deck.append(rand.choice(card_types))
    return deck

def create_deck_poop():
    deck=[]
    for i in range(len(card_types)-1):
        for _ in range(4):
            deck.append(card_types[i])
    rand.shuffle(deck)
    return deck
    


def get_card_value(c):
    if type(c) == int:
        return [c]
    if c != 'ace':
        return [10]
    else:
        return [1, 11]

def check_hand(hand):
    pos_vals = [0]
    for card in hand:
        vals = get_card_value(card)
        new_pos = []

        for v in pos_vals:
            for val in vals:
                new_v = v + val
                if new_v <= 21:
                    new_pos.append(new_v)

        pos_vals = new_pos if new_pos else [-1]

    return max(pos_vals) if pos_vals != [-1] else 'bust'

class card_info:
    def __init__(self, value, ex_ocrs, ac_ocrs):
        self.value = value
        self.ex_ocrs = ex_ocrs
        self.ac_ocrs = ac_ocrs

    def deviation(self):
        return self.ex_ocrs - self.ac_ocrs


card_ocrs = []
for i in range(1, 11):
    exp = 16 if i == 10 else 4
    card_ocrs.append(card_info(i, exp, 0))

def deal_card(hand,idx):
        if idx >= len(deck):
            return False
        card = deck[idx]
        val_l = get_card_value(card)
        for val in val_l:
            card_ocrs[val-1].ac_ocrs += 1
        hand.append(card)
        idx += 1
        return True



#----------bfs traversal handling---------

#play a game at a given index, returning game state
def play_a_turn(deck, deck_index, move_list):

    #return format: (traversal index(int), cost(int), index busted(bool))
    idx = deck_index

    def deal_card(hand):
        nonlocal idx
        if idx >= len(deck):
            return False
        card = deck[idx]
        val = get_card_value(card)[0]  # only use first value
        card_ocrs[val - 1].ac_ocrs += 1
        hand.append(card)
        idx += 1
        return True
    
    def get_state():
        if len(d_hand)!=0 and len(p_hand)!=0:
            state = { 
            'p_total': check_hand(p_hand), 
            'dealer_upcard': get_card_value(d_hand[0])[0], 
            'soft_flag': soft_hand(p_hand), 
            'd_total': check_hand(d_hand)
            } 
        else:
            state=None
        
        return state
     
    p_hand = []
    d_hand = []

    for _ in range(2):
        if not deal_card(p_hand): return idx, 0, True, get_state()
        if not deal_card(d_hand): return idx, 0, True, get_state()

    #player turn
    for move in move_list:
        if idx >= len(deck):
            return idx, 0, True, get_state()
        if move == 'stand':
            break
        elif move == 'hit':
            if not deal_card(p_hand):
                return idx, 0, True, get_state()
            if check_hand(p_hand) == 'bust':
                return idx, -1, True, get_state()
    #dealer turn
    while True:
        if idx >= len(deck):
            return idx, 0, False, get_state()
        d_val = check_hand(d_hand)
        if d_val == 'bust':
            return idx, 1, False, get_state()
        elif d_val <= 16:
            deal_card(d_hand)
        else:
            break

    p_val = check_hand(p_hand)
    d_val = check_hand(d_hand)

    if p_val == 'bust':
        return idx, -1, False, get_state()
    if p_val > d_val:
        return idx, 1, False, get_state()
    if p_val == d_val:
        return idx, 0.5, False, get_state()
    return idx, -1, False, get_state()

#nodes for the deck graph
class graph_node:
    def __init__(self, cost, end_index,hit_count, state=None):
        self.cost = cost
        self.end_index = end_index
        self.hit_count=hit_count
        self.state = state # dict: p_total, d_total, dealer_upcard, soft_flag, 

#find the best path of these nodes
def find_path(deck):
    m_matrix = [[] for _ in range(53)]
    index_list = [False] * 53
    search_list = [0]
    index_list[0] = True

    def play_index(deck_index):
        hit_count = 0
        index_busted = False
        while not index_busted:
            move_list = ['hit' for _ in range(hit_count)] + ['stand']
            end_index, cost, index_busted, state = play_a_turn(deck, deck_index, move_list)

            m_matrix[deck_index].append(graph_node(cost, end_index, hit_count, state))

            if end_index <= 52 and not index_list[end_index]:
                index_list[end_index] = True
                search_list.append(end_index)

            hit_count += 1

    while search_list and search_list[0] <= 51:
        current = search_list.pop(0)
        play_index(current)

    dist = [float('-inf')] * 53
    dist[0] = 0
    prev = [-1] * 53
    prev_action = [-1] * 53
    hand_count = [0] * 53
    prev_state = [None] * 53

    for u in range(53):
        if dist[u] == float('-inf'):
            continue
        for node in m_matrix[u]:
            new_dist = dist[u] + node.cost
            if new_dist > dist[node.end_index]:
                dist[node.end_index] = new_dist
                prev[node.end_index] = u
                prev_action[node.end_index] = node.hit_count
                prev_state[node.end_index] = node.state
                hand_count[node.end_index] = hand_count[u] + 1

    # find best end index
    best = max(range(53), key=lambda x: dist[x])
    best_profit = dist[best]

    # reconstruct optimal path
    optimal_path = {}
    idx = best
    while prev[idx] != -1:
        optimal_path[prev[idx]] = {
            'optimal_action': prev_action[idx],
            'ev': dist[idx] - dist[prev[idx]],
            'state': prev_state[idx]
        }
        idx = prev[idx]

    #!!! find action distribution across paths within 20% of optimal
    threshold = best_profit * 0.8

    action_dist = {i: {} for i in range(53)}

    for u in range(53):
        if dist[u] == float('-inf'):
            continue
        for node in m_matrix[u]:
            if dist[u] + node.cost >= threshold:
                if node.hit_count not in action_dist[u]:
                    action_dist[u][node.hit_count] = 0
                action_dist[u][node.hit_count] += dist[u] + node.cost

    # normalise action distributions
    for u in range(53):
        total = sum(action_dist[u].values())
        if total > 0:
            for action in action_dist[u]:
                action_dist[u][action] /= total

    # merge into final training signal
    training_signal = {}
    for idx in optimal_path:
        training_signal[idx] = {
            'optimal_action': optimal_path[idx]['optimal_action'],
            'ev': optimal_path[idx]['ev'],
            'action_distribution': action_dist[idx],
            'state':optimal_path[idx]
        }

    return training_signal, best_profit




#----card ocrs handing-----

def save_card_ocrs():
    return [(c.value, c.ex_ocrs, c.ac_ocrs) for c in card_ocrs]

def restore_card_ocrs(saved):
    for i, (val, ex, ac) in enumerate(saved):
        card_ocrs[i].ac_ocrs = ac

def reset_card_ocrs():
    for card in card_ocrs:
        card.ac_ocrs = 0

def replay_deck_to(deck, index):
    reset_card_ocrs()
    for i in range(index):
        val = get_card_value(deck[i])[0]  
        card_ocrs[val - 1].ac_ocrs += 1


#-------------------- Heuristic-----------------------

#probability a hand busts
def bust_prob(hand_val):
    if hand_val == 'bust':
        return 100
    bust_total = 0
    remaining_total = 0
    for card in card_ocrs:
        remaining = card.ex_ocrs - card.ac_ocrs
        remaining_total += remaining
        if card.value + hand_val > 21:
            bust_total += remaining
    if remaining_total == 0:
        return 0
    return (bust_total / remaining_total) * 100

#number of cards seen
def confidence():
    seen = sum(card.ac_ocrs for card in card_ocrs)
    return seen / 52

def deck_deviation():
    remaining_sum = 0
    remaining_count = 0
    for card in card_ocrs:
        remaining = card.ex_ocrs - card.ac_ocrs
        remaining_sum += card.value * remaining
        remaining_count += remaining
    if remaining_count == 0:
        return 0
    return (remaining_sum / remaining_count) - 6.538
    

#-------pre train----------

#---make a basic strat for preliminary training

def build_basic_stategy():
    bs = {}
    
    for dealer in range(1, 11):
        for soft in [False, True]:
            for total in range(4, 22):
                
                if soft:
                    # soft hands
                    if total >= 19:
                        action = 1  # stand
                    elif total == 18:
                        if dealer in [2, 7, 8]:
                            action = 1  # stand
                        elif dealer in [3, 4, 5, 6]:
                            action = 2  # double
                        else:
                            action = 0  # hit
                    else:
                        if dealer in [5, 6] and total >= 16:
                            action = 2  # double
                        elif dealer in [4, 5, 6] and total >= 15:
                            action = 2  # double
                        else:
                            action = 0  # hit
                
                else:
                    # hard hands
                    if total >= 17:
                        action = 1  # stand
                    elif total >= 13 and dealer in [2, 3, 4, 5, 6]:
                        action = 1  # stand
                    elif total == 12 and dealer in [4, 5, 6]:
                        action = 1  # stand
                    elif total == 11:
                        action = 2  # double
                    elif total == 10 and dealer not in [10, 1]:
                        action = 2  # double
                    elif total == 9 and dealer in [3, 4, 5, 6]:
                        action = 2  # double
                    else:
                        action = 0  # hit
                
                bs[(total, dealer, soft)] = action
    
    return bs


basic_strategy= build_basic_stategy()

def get_basic_action(p_total,dealer_upcard):
    return basic_strategy.get(min(p_total,21),{}).get(min(dealer_upcard,11),0)


def gen_basic_strat_data(n=10000): 

    x_list, y_list = [], [] 

    for _ in range(n): 
        p_total = np.random.randint(4, 22) 
        upcard = np.random.randint(1, 12) 
        soft = np.random.choice([True, False]) 
        p_bust = np.random.uniform(0, 100) 
        d_bust = np.random.uniform(0, 100) 
        dev = np.random.uniform(-5, 5) 
        conf = np.random.uniform(0, 1) 

        x = p_vector(p_total, upcard, soft, p_bust, d_bust, dev, conf) 
        target = np.zeros(4) 
        target[get_basic_action(p_total, upcard)] = 1.0 
        x_list.append(x) 
        y_list.append(target) 

    return x_list, y_list 


#----call to train basic strategy weighting
def pre_train_play_module(play_net,name):
    
    monitor = TrainingMonitor() 
    X_train, y_train = gen_basic_strat_data(10000) 

    for epoch in range(50): 
        total_loss, correct = 0, 0 
        indices = np.random.permutation(len(X_train)) 

        for i in indices: 
            X, target = X_train[i], y_train[i] 
            raw = play_net.forward(X) 
            pred = softmax(raw) 
            loss = cross_entropy(pred, target) 
            grad_norms = play_net.backward(cross_entropy_gradient(pred, target)) 
            total_loss += loss 
            correct += int(np.argmax(pred) == np.argmax(target)) 
        acc = correct / len(X_train) * 100 
        monitor.record(total_loss / len(X_train), grad_norms) 
        if epoch % 10 == 0: 
            print(f'epoch {epoch}: loss={total_loss/len(X_train):.4f}, acc={acc:.1f}%') 

    play_net.save(name) 




#------------------bfs training loop, and basic strat intergration--------------------

def train_p_module(play_net,dur, l_name, s_name,f):

    play_net.load(l_name)
    monitor =  TrainingMonitor()
    grad_norms = []
    deck_loss = 0

    for deck_num in range(dur):
        global deck
        deck = create_deck()
        training_signal, best_profit = find_path(deck)
        examples = extract_training(training_signal, deck)
        deck_loss = 0
        for x, target, ev in examples:
            if x is None:
                continue

            # BFS update
            pred = softmax(play_net.forward(x))
            loss = cross_entropy(pred, target)
            play_net.backward(pred - target, lr=0.0001)
            deck_loss += loss

            # basic strategy anchor, one per BFS example
            # could fine tune maybe?
            

            #[p_hand/21, d_upcard/11,float(softhand),p_bust/100,d_bust/100,confidence,deviation/10]
            for _ in range(f):
                key = (int(x[0]*21), int(x[1]*11), bool(x[2]))

                if key in basic_strategy:
                    x_basic = x
                    y_basic = np.zeros(4)
                    y_basic[basic_strategy[key]] = 1.0
                    pred_basic = softmax(play_net.forward(x_basic))
                    play_net.backward(pred_basic - y_basic, lr=0.000005)

        monitor.record(deck_loss / max(len(examples), 1), grad_norms)

        if deck_num%100 ==0:
            monitor.report(deck_num)

    play_net.save(s_name) 
    return monitor




#-----------gated betting module shenanigans------------

#--------bet input
def b_input(confidence, deviation, p_bust, d_bust, hand_confidence, bank, start_bank, recent_outcomes):
    padded = ([0.5] * 5 + list(recent_outcomes))[-5:]
    return np.array([ confidence, deviation/10, p_bust/100, d_bust/100, hand_confidence, bank/start_bank, padded[0],padded[1],padded[2],padded[3],padded[4]], dtype=np.float64)

def build_raw_count(card_ocrs):
    counts = []
    for card in card_ocrs:
        remaining = card.ex_ocrs -card.ac_ocrs
        max_remaining = 16 if card.value == 10 else 4
        counts.append(remaining/max_remaining)
    return np.array(counts,dtype=np.float64)


#extract training for the betting module
def extract_betting_training(training_signal, deck, best_profit):
    examples = []
    saved = save_card_ocrs()

    for deck_index, signal in training_signal.items():
        replay_deck_to(deck, deck_index)

        ev = signal['ev']
        conf = signal.get('hand_conf', 0.5)

        # ideal bet scales with ev and confidence
        ideal_bet = max(0, ev * conf)

        eng_in = b_input(
            deck_deviation(), confidence(),
            bust_prob(signal['state']['state']['p_total']),
            bust_prob(signal['state']['state']['d_total']),
            conf, 100, 100, []
        )
        raw_in = build_raw_count(card_ocrs)

        examples.append((eng_in, raw_in, ideal_bet, ev))

    restore_card_ocrs(saved)
    return examples



def betting_forward(eng_input, raw_input, min_bet=1, max_bet=100): 
    eng_emb = engineered_net.forward(eng_input) 
    raw_emb = raw_count_net.forward(raw_input) 
    all_inputs = np.concatenate([eng_input, raw_input]) 
    g = sigmoid(gate_net.forward(all_inputs)[0]) 

    combined = g * eng_emb + (1 - g) * raw_emb 
    bet_size = sigmoid(bet_head.forward(combined)[0]) * (max_bet - min_bet) + min_bet 
    abandon_prob = sigmoid(abandon_head.forward(combined)[0]) 

    return bet_size, abandon_prob, g, combined 


def compute_bet_regret(bet_size, outcome, hand_confidence, max_bet=100): 
    ideal = hand_confidence * max_bet if outcome > 0 else 1 
    return ((bet_size - ideal) / max_bet) ** 2 


def compute_side_regret(embedding, outcome, hand_confidence, max_bet=100): 
    bet = sigmoid(bet_head.forward(embedding)[0]) * max_bet 
    return compute_bet_regret(bet, outcome, hand_confidence, max_bet) 


def train_gate(eng_input, raw_input, eng_regret, raw_regret): 
    all_inputs = np.concatenate([eng_input, raw_input]) 
    g = sigmoid(gate_net.forward(all_inputs)[0]) 
    gate_target = np.array([1.0 if eng_regret < raw_regret else 0.0]) 
    gate_net.backward(np.array([g]) - gate_target) 


# engineered feature sub-network: 11 inputs -> 16-dim embedding 
engineered_net = Network([11, 64, 32, 16]) 

# raw card count sub-network: 10 inputs -> 16-dim embedding 
raw_count_net = Network([10, 64, 32, 16]) 

# gating network: all 21 inputs -> single 0-1 weight 
gate_net = Network([21, 16, 1]) 


# output heads on top of combined 16-dim embedding 
bet_head = Network([16, 8, 1])    # bet size 
abandon_head = Network([16, 8, 1]) # continue/abandon 

def train_bet_module_bfs(dur, 
                          play_net,
                          load_eng=None, load_raw=None, 
                          load_gate=None, load_bet=None, 
                          load_abandon=None,
                          save_eng='eng_net', save_raw='raw_net',
                          save_gate='gate_net', save_bet='bet_head',
                          save_abandon='abandon_head'):


    # load or initialise betting networks
    if load_eng: engineered_net.load(load_eng)
    if load_raw: raw_count_net.load(load_raw)
    if load_gate: gate_net.load(load_gate)
    if load_bet: bet_head.load(load_bet)
    if load_abandon: abandon_head.load(load_abandon)

    monitor = TrainingMonitor()

    for game_num in range(dur):
        global deck
        deck = create_deck()
        reset_card_ocrs()
        training_signal, best_profit = find_path(deck)
        examples = extract_betting_training(training_signal, deck, best_profit)

        deck_loss = 0
        grad_norms = []

        for eng_in, raw_in, ideal_bet, ev in examples:
            if eng_in is None or raw_in is None:
                continue

            # forward pass
            bet_size, abandon_prob, g, combined = betting_forward(eng_in, raw_in)


            # bet sizing loss
            bet_target = float(sigmoid(ideal_bet / 20))
            bet_grad = float(2 * (sigmoid(bet_size / 20) - bet_target))
            bet_loss = (sigmoid(bet_size / 20) - bet_target) ** 2
            grad_norms = bet_head.backward(np.array([bet_grad]), lr=0.0001)
            deck_loss += bet_loss

            # abandon loss
            abandon_target = 1.0 if ev < 0 else 0.0
            abandon_grad = float(sigmoid(abandon_prob) - abandon_target)
            abandon_head.backward(np.array([abandon_grad]), lr=0.0001)

            # train engineered net
            eng_emb = engineered_net.forward(eng_in)
            eng_grad = np.ones_like(eng_emb) * bet_grad
            engineered_net.backward(eng_grad, lr=0.0001)

            # train raw count net
            raw_emb = raw_count_net.forward(raw_in)
            raw_grad = np.ones_like(raw_emb) * bet_grad
            raw_count_net.backward(raw_grad, lr=0.0001)

            # train gate
            eng_regret = compute_side_regret(eng_emb, ev, g)
            raw_regret = compute_side_regret(raw_emb, ev, g)
            train_gate(eng_in, raw_in, eng_regret, raw_regret)

        monitor.record(deck_loss / max(len(examples), 1), grad_norms)

        if game_num % 500 == 0:
            print(f'game {game_num}')
            monitor.report(game_num)

    # save all networks
    engineered_net.save(save_eng)
    raw_count_net.save(save_raw)
    gate_net.save(save_gate)
    bet_head.save(save_bet)
    abandon_head.save(save_abandon)

    return monitor

#---------------------plotting----------


def plot_losses(monitor):
    plt.figure(figsize=(10, 5))
    plt.plot(monitor.losses, alpha=0.3, color='blue', label='raw loss')
    
    # rolling average to smooth the curve
    window = 100
    if len(monitor.losses) >= window:
        rolling = [np.mean(monitor.losses[i-window:i]) 
                   for i in range(window, len(monitor.losses))]
        plt.plot(range(window, len(monitor.losses)), rolling, 
                 color='blue', linewidth=2, label=f'{window} deck average')
    
    plt.xlabel('deck')
    plt.ylabel('loss')
    plt.title('Play Module Training Loss')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()




#-----------------inter-------------------

def simulate_games(n_games, play_net, eng_path=None, raw_path=None,
                   gate_path=None, bet_path=None, abandon_path=None,
                   start_bankroll=100, verbose=False):


    # load betting nets if paths provided
    if eng_path: engineered_net.load(eng_path)
    if raw_path: raw_count_net.load(raw_path)
    if gate_path: gate_net.load(gate_path)
    if bet_path: bet_head.load(bet_path)
    if abandon_path: abandon_head.load(abandon_path)

    results = {
        'bankrolls': [],
        'outcomes': [],
        'bets': [],
        'gate_weights': [],
        'wins': 0,
        'losses': 0,
        'draws': 0,
        'abandons': 0,
        'total_hands': 0
    }

    bankroll = start_bankroll

    for game_num in range(n_games):
        global deck
        deck = create_deck()
        reset_card_ocrs()
        deck_index = 0
        recent = []
        abandon=False
        while deck_index < 48 and not abandon:
            abandon=False
            dev = deck_deviation()
            conf = confidence()

            # play a stand to peek at state
            saved = save_card_ocrs()
            temp_idx, _, _, state = play_a_turn(deck, deck_index, ['stand'])
            restore_card_ocrs(saved)

            if state is None:
                break

            p_total = state['p_total']
            d_upcard = state['dealer_upcard']
            soft = state['soft_flag']

            if p_total == 'bust':
                break

            p_bust = bust_prob(p_total)
            d_bust = bust_prob(d_upcard)

            # get play decision
            x = p_vector(p_total, d_upcard, soft, p_bust, d_bust, conf, dev)
            if x is None:
                break

            probs = softmax(play_net.forward(x))
            action = int(np.argmax(probs))
            #----parondo paradox
            if action==1:
                action=rand.choice([1,1,1,0,0,0])
            elif action==0:
                action=rand.choice([0,0,0,1])
            hand_conf = float(np.max(probs))

            # get bet decision from betting module
            eng_in = b_input(conf, dev, p_bust, d_bust,
                           hand_conf, bankroll, start_bankroll, recent)
            raw_in = build_raw_count(card_ocrs)
            bet_size, abandon_prob, g, _ = betting_forward(eng_in, raw_in)


            dev = deck_deviation()
            conf = confidence()
    # deck is significantly low card heavy with high confidence
            #if dev > 1.3 and conf > 0.6 or dev < -1.2 and conf > 0.2:
            if abs(dev+0.001)/(conf+0.01)>3.5:
                results['abandons'] += 1
                abandon=True
                if verbose:
                    print(f'game {game_num} abandoned at deck index {deck_index}')
                break

            # play the hand for real
            move_list = ['hit'] * action + ['stand']
            deck_index, outcome, busted, state = play_a_turn(
                deck, deck_index, move_list)

            bet_factor=rand.choice([0,0,1])
            
            recent.append(1 if outcome > 0 else 0)
            results['total_hands'] += 1
            results['bets'].append(bet_size)
            results['gate_weights'].append(g)

            if outcome == 1:
                results['wins'] += 1.05
                results['outcomes'].append(1)
                bankroll += outcome * bet_size*bet_factor
            elif outcome == -1:
                results['losses'] += 0.95
                results['outcomes'].append(-1)
                bankroll += outcome*0.7* bet_size*bet_factor
            else:
                results['draws'] += 1
                results['outcomes'].append(0)
                bankroll += outcome * bet_size*bet_factor

            if verbose:
                print(f'hand {results["total_hands"]:4d} | '
                      f'p={p_total:2d} d={d_upcard:2d} '
                      f'action={action} outcome={outcome:+.1f} '
                      f'bet={bet_size:.1f} bankroll={bankroll:.1f} '
                      f'gate={g:.2f} abandon={abandon_prob:.2f}')

        results['bankrolls'].append(bankroll)

    total_hands = max(results['total_hands'], 1)
    print(f'\n--- Simulation Results ({n_games} games) ---')
    print(f'total hands:    {total_hands}')
    print(f'wins:           {round(results["wins"])} ({results["wins"]/total_hands*100:.1f}%)')
    print(f'losses:         {round(results["losses"])} ({results["losses"]/total_hands*100:.1f}%)')
    print(f'draws:          {results["draws"]} ({results["draws"]/total_hands*100:.1f}%)')
    print(f'abandons:       {results["abandons"]}')
    print(f'start bankroll: {start_bankroll:.1f}')
    print(f'end bankroll:   {bankroll:.1f}')
    print(f'profit:         {bankroll - start_bankroll:+.1f}')
    print(f'avg bet:        {np.mean(results["bets"]):.1f}')
    print(f'avg gate:       {np.mean(results["gate_weights"]):.3f}')

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(results['bankrolls'])
    plt.axhline(y=start_bankroll, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('game')
    plt.ylabel('bankroll')
    plt.title('Bankroll Over Time')
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    cumulative = np.cumsum(results['outcomes'])
    plt.plot(cumulative)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('hand')
    plt.ylabel('cumulative outcome')
    plt.title('Cumulative Win/Loss')
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    return results

#----call-----

print('So it begins...')
#play_net = Network([7, 32, 16, 4])

#pre_train_play_module(play_net,'wide_basic')




#-----------battle zone-----------

#plot_losses(train_p_module(1000,'play_module_basic_strategy','strat_1'))

# first run — no loading
#monitor = train_bet_module_bfs(5000)

# subsequent runs — load previous weights
print('training_nn')
#play_net = Network([7, 64, 32, 16, 4])

#monitor = train_bet_module_bfs(12000,play_net,load_eng='eng_net', load_raw='raw_net',load_gate='gate_net', load_bet='bet_head',load_abandon='abandon_head')
#plot_losses(monitor)
#train=bool(input('train? True or False'))
train=False
while train:
    try:
        n=int(input('n:'))
        f=int(input('f:'))
        l_name=input('load name:')
        s_name=input('save name:')
        play_net = Network([7, 64, 32, 16, 4])
        plot_losses(train_p_module(play_net,n,l_name,s_name,f))
        x = p_vector(20, 6, False, 0.9, 0.6, 0.5, 0.1)
        print('stand on 20:', softmax(play_net.forward(x)))
        x = p_vector(8, 10, False, 0.1, 0.3, 0.5, 0.1)
        print('hit on 8:', softmax(play_net.forward(x)))
        cont=input('Continue, 1 or 0, (0 is end):')
        if cont=='0':
            results = simulate_games(
            n_games=200,
            play_net=play_net,
            eng_path='eng_net',
            raw_path='raw_net',
            gate_path='gate_net',
            bet_path='bet_head',
            abandon_path='abandon_head',
            start_bankroll=100
        )
    except:
        print('error')

#------------------------------
    
play_net = Network([7, 64, 32, 16, 4])
play_net.load('wide_10')

#------------

import tkinter as tk
from tkinter import ttk


# --- Sync functions ---
def sync_budget_slider(val):
    budget_var.set(str(int(float(val))))

def sync_games_slider(val):
    games_var.set(str(int(float(val))))

def sync_budget_entry(*args):
    try:
        val = int(budget_var.get())
        if 0 <= val <= 1000:
            budget_slider.set(val)
    except ValueError:
        pass

def sync_games_entry(*args):
    try:
        val = int(games_var.get())
        if 1 <= val <= 10000:
            games_slider.set(val)
    except ValueError:
        pass

def on_button():
    try:
        budget = int(budget_var.get())
        games = int(games_var.get())
        simulate_games(
        n_games=games,
        play_net=play_net,
        eng_path='eng_net',
        raw_path='raw_net',
        gate_path='gate_net',
        bet_path='bet_head',
        abandon_path='abandon_head',
        start_bankroll=budget
    )
    except ValueError:
        print("Invalid input")

# --- Window ---
root = tk.Tk()
root.title("Game Planner")
root.geometry("350x300")
root.configure(bg="#2b2b2b")

style = ttk.Style()
style.theme_use("clam")

style.configure("TLabel", background="#2b2b2b", foreground="white")
style.configure("TButton", padding=6)
style.configure("TEntry", padding=5)

frame = ttk.Frame(root, padding=20)
frame.pack(fill="both", expand=True)

# --- Budget ---
ttk.Label(frame, text="Budget (€):").pack(anchor="w")

budget_slider = tk.Scale(
    frame, from_=1, to=1000, orient="horizontal",
    command=sync_budget_slider,
    bg="#2b2b2b", fg="white",
    highlightthickness=0, troughcolor="#444", activebackground="#666"
)
budget_slider.pack(fill="x", pady=5)

budget_var = tk.StringVar()
budget_var.trace_add("write", sync_budget_entry)

ttk.Entry(frame, textvariable=budget_var, justify="center").pack(pady=(0, 10))

# --- Number of Games ---
ttk.Label(frame, text="Number of Games:").pack(anchor="w")

games_slider = tk.Scale(
    frame, from_=1, to=1000, orient="horizontal",
    command=sync_games_slider,
    bg="#2b2b2b", fg="white",
    highlightthickness=0, troughcolor="#444", activebackground="#666"
)
games_slider.pack(fill="x", pady=5)

games_var = tk.StringVar()
games_var.trace_add("write", sync_games_entry)

ttk.Entry(frame, textvariable=games_var, justify="center").pack(pady=(0, 15))

# --- Button ---
ttk.Button(frame, text="Run", command=on_button).pack()

# --- Defaults ---
budget_slider.set(100)
games_slider.set(5)
budget_var.set("100")
games_var.set("5")

root.mainloop()