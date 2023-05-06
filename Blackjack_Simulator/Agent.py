import random
from pprint import pprint
from copy import deepcopy

LOSS = -100
DRAW = 0
WIN = 100
STAY = " "
HIT = "H"

random.seed(340)


class Agent:
    def __init__(self, learning_rate, exploration_rate, iterations):
        # initialize base attributes
        self.learningRate = learning_rate
        self.explorationRate = exploration_rate
        self.iterations = iterations

        self.actions = [0, 1]  # 0 for STAND, 1 for HIT
        self.state = (0, 0, False)  # (playerHand, dealerFaceUpCard, usableAce)
        self.deck = Deck()

        self.playerCards = []
        self.dealerCards = []

        # initialize Q-values
        self.QValues = {}  # key : tuple of playerHand, dealerFaceUpCard and usableAce | value : dictionary:{key: action, value: Q-value}
        for playerCard in range(4, 22):
            for dealerFaceUpCard in range(2, 12):
                for usableAce in [True, False]:
                    self.QValues[(playerCard, dealerFaceUpCard,
                                  usableAce)] = {}
                    for action in self.actions:
                        if playerCard == 21 and action == 0:
                            self.QValues[(
                                playerCard, dealerFaceUpCard, usableAce)][action] = WIN
                        else:
                            self.QValues[(
                                playerCard, dealerFaceUpCard, usableAce)][action] = DRAW

    # initialize rewards
    # self.rewards = deepcopy(self.Q_values)

    def addCardValue(self, card, hand, usableAce):
        if card[1] == 'A':
            if hand + 11 > 21:
                hand += 1
            else:
                hand += 11
                usableAce = True
        else:
            hand += card[1]
        
        return hand, usableAce

    # Dealer policy function
    def dealerPolicy(self, dealerCards):

        # Total up dealer's cards
        dealerHand = 0
        usableAce = False
        for card in dealerCards:
            dealerHand, usableAce = self.addCardValue(card, dealerHand, usableAce)

        # Check if dealer can still draw
        if dealerHand >= 17:                                # Dealer value is greater than 17, check if it goes over 21
            if dealerHand > 21 and usableAce and dealerHand - 10 < 17:                                 # Dealer value is greater than 21
                dealerHand -= 10                                # Turn Ace into 1
                usableAce = False                               # Set usable Ace to False
                card = self.deck.drawCard()                         # Draw card
                dealerHand, usableAce = self.addCardValue(card, dealerHand, usableAce)

        else:                                               # Dealer value is less than 17
            card = self.deck.drawCard()                         # Draw card
            dealerHand, usableAce = self.addCardValue(card, dealerHand, usableAce)

            dealerCards.append(card)

        # Return new value and usableAce
        return dealerHand, usableAce

    def chooseAction(self):
        agentShouldExplore = random.uniform(0, 1) <= self.explorationRate

        if agentShouldExplore:
            return self.actions[random.randint(0, 1)]
        else:
            bestAction = max(
                self.QValues[self.state], key=self.QValues[self.state].get)
            return bestAction

    # need to modify this to fit this implementation. maybe rename
    # input: current state, chosenAction
    # output: next state
    def getNextState(self, chosenAction):
        # check action, if it's hit add new card, else don't
        # add new card to current state and update accordingly
        # consider usableAce too
        # print('in getNextState')
        # print('chosenAction=', chosenAction)
        playerHand = self.state[0]
        dealerFaceUpCard = self.state[1]
        usableAce = self.state[2]

        if chosenAction == 1:         # If chosenAction == HIT....
            newCard = self.deck.drawCard()
            self.playerCards.append(newCard)
            playerHand, usableAce = self.addCardValue(newCard, playerHand, usableAce)

        nextState = (playerHand, dealerFaceUpCard, usableAce)
        return nextState

    # slightly modified from DS340 RL assignment
    # this function should run everytime the agent does an action
    def newQValue(self, currentState, nextState, action):
        if nextState[0] > 21:
            currentQValue = self.QValues[currentState][action]
            nextQValue = LOSS
            newQValue = currentQValue + self.learningRate * \
                (nextQValue - currentQValue)
        else:
            currentQValue = self.QValues[currentState][action]
            nextQValue = max(self.QValues[nextState].values())
            newQValue = currentQValue + self.learningRate * \
                (nextQValue - currentQValue)

        return newQValue

# Needs to have three states: player wins, draw, dealerwins.
# reward function
    def checkWinner(self, playerHand, dealerHand):
        playerBlackJack = False
        dealerBlackJack = False

        if playerHand == 21 and len(self.playerCards) == 2:
            playerBlackJack = True
        if dealerHand == 21 and len(self.dealerCards) == 2:
            dealerBlackJack = True

        if playerHand > 21 or (not playerBlackJack and dealerBlackJack) or ((playerHand < dealerHand[0]) and dealerHand[0] <= 21):
            return LOSS
        elif (playerBlackJack and dealerBlackJack) or playerHand == dealerHand:
            return DRAW
        else:
            return WIN

    # returns empty policy map
    # totalMap[usableAce][playerHand][dealerFaceUpCard]
    def getPolicy(self):
        totalMap = []
        map = []
        row = [" "] * 10
        for _ in range(2):
            for _ in range(18):
                map.append(deepcopy(row))
            totalMap.append(deepcopy(map))
        return totalMap

    def play(self):
        policy = {}
        for _iter in range(self.iterations):
            # RESET GAME
            self.state = (0, 0, False)
            self.playerCards = []
            self.dealerCards = []
            self.deck.replenish()

            # deal cards to agent and dealer
            self.playerCards = self.deck.drawCards(2)
            self.dealerCards = self.deck.drawCards(2)

            # update state after cards dealt
            dealerFaceUpCard = 11 if self.dealerCards[0][1] == "A" else self.dealerCards[0][1]
            usableAce = False
            playerHand = 0

            for card in self.playerCards:
                playerHand, usableAce = self.addCardValue(card, playerHand, usableAce)

            self.state = (playerHand, dealerFaceUpCard, usableAce)
            # print('starting state for iteration', _iter, "=", self.state)

            bust = False
            while True:
                action = self.chooseAction()
                # print('action rec=', HIT if action == 1 else STAY)
                if action == 0: # if stay, get out of loop
                    break
                nextState = self.getNextState(action)
                currQVal = self.QValues[self.state]
                # print('nextState=', nextState)
                # print('pre-update Q=', self.QValues[self.state][action])
                self.QValues[self.state][action] = self.newQValue(self.state, nextState, action)
                policy[self.state] = HIT if max(currQVal, key= lambda x: currQVal[x]) == 1 else STAY
                # print('post-update Q=', self.QValues[self.state][action])
                # print()
                if nextState[0] > 21:
                    bust = True
                    break
                self.state = nextState
                    

            # continues to next iteration if we bust
            if bust:
                continue

            # run dealerPolicy
            finalDealerHand = self.dealerPolicy(self.dealerCards)

            # check winner
            currentQValue = self.QValues[self.state][action]
            reward = currentQValue + self.learningRate * \
                (self.checkWinner(self.state[0], finalDealerHand) - currentQValue)
            self.QValues[self.state][action] = reward
        # pprint(self.QValues)
        for k in self.QValues:
            if k[0] < 11 and k[2] == True: # disregards impossible states (playerHand < 11 and usableAce == True)
                continue
            currValue = self.QValues[k]
            policy[k] = HIT if max(currValue, key= lambda x: currValue[x]) == 1 else STAY
        return policy


# Right now, we are operating under the assumption that there is only one deck.
# Possible expansion: adding additional decks to decrease effectiveness of card counting (if implemented)
class Deck:
    def __init__(self):
        self.deck = []
        c = 'cdhs'    # clubs, diamonds, hearts, spades
        values = ['A', 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]

        for i in range(4):
            suit = [[c[i], v] for v in values]
            self.deck.extend(suit)

    def drawCard(self):
        cardIdx = random.randrange(len(self.deck))
        ret_card = self.deck.pop(cardIdx)
        return ret_card

    def drawCards(self, card_count):
        cards = []
        for _ in range(card_count):
            cards.append(self.drawCard())

        return cards

    def print(self):
        pprint(self.deck)

    def length(self):
        return len(self.deck)

    def replenish(self):
        self.__init__()
        return True

agent = Agent(0.5, 0.5, 100000)

policy = agent.play()

# pprint(policy)

# false_dict = {}
# true_dict = {}
f_arr = [[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0]]
t_arr = [[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0]]
f_hc = 0
t_hc = 0

# Loop through each key-value pair in the original dictionary
for key, value in policy.items():
    # print(key)
    playerHand = key[0]
    dealerVis = key[1]
    usableAce = key[2]
    # Check the value of the third index in the key
    if playerHand < 13:
        continue
    if usableAce == False:
        # Add the key-value pair to the false dictionary
        # false_dict[key] = value
        f_arr[playerHand - 13][dealerVis - 2] = value
        f_hc += 1 if value == HIT else 0
    else:
        # Add the key-value pair to the true dictionary
        # true_dict[key] = value
        t_arr[playerHand - 13][dealerVis - 2] = value
        t_hc += 1 if value == HIT else 0

# pprint(false_dict)
# print()
# pprint(true_dict)
# print('\n\n')
pprint(t_arr)
print()
pprint(f_arr)

# print('f_hc=', f_hc)
# print('f_hc ideal= 25')
# print('t_hc=', t_hc)
# print('f_hc ideal= 50')