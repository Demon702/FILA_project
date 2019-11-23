import numpy as np
import pickle
import pygame   
import sys, pdb
import argparse
from model import Model
import torch
from torchvision import transforms
import torch.optim as optim
from torch.optim import lr_scheduler
import copy
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class State:
	def __init__(self, p1, p2 , size = 10):
		self.win_reward = 5
		self.size = size
		self.total_squares = (size - 1)**2
		self.board = np.zeros((size, size - 1 , 2) , dtype = np.int32)
		self.old_board = np.zeros((size, size - 1 , 2) , dtype = np.int32)
		self.p1 = p1
		self.p2 = p2
		self.isEnd = False
		self.boardHash = None
		# init p1 plays first
		# self.playerSymbol = 1
		self.no_of_boards = 0
		self.grid_winners = np.zeros((size - 1, size - 1) , dtype = np.int32)
		self.loss = torch.nn.MSELoss()
		self.winners = []

	def getHash(self):
		string = str(self.board.reshape((size - 1) * size * 2))
		self.boardHash = int(string , 2)
		return self.boardHash

	def tuple_to_number(self , index ):
		return 2*(self.size - 1)*index[0] + 2*index[1] + index[2]

	def number_to_tuple(self , number):
		return [number//(2*(self.size - 1)) , (number % (2*(self.size - 1)))//2 , number % 2]

	def reset(self):
		self.board = np.zeros((size, size - 1 , 2) , dtype = np.int32)
		# self.playerSymbol = 1
		self.no_of_boards = 0
		self.isEnd = False
		self.grid_winners = np.zeros((size - 1, size - 1) , dtype = np.int32)

	def available_actions(self):
		available_array = np.where(self.board == 0)
		available_array = np.array(np.transpose(available_array))
		available_actions = [2*(self.size - 1)*index[0] + 2*index[1] + index[2] for index in available_array]
		return available_actions

	def updatestate(self, action):
		[x , y , z] = [action//(2*(self.size - 1)) , (action % (2*(self.size - 1)))//2 , action % 2]
		self.board[x , y, z] = 1

	def winner(self):
		# print("P1 has won :" , sum(sum(self.grid_winners == 1)))
		# print("P2 has won :" , sum(sum(self.grid_winners == -1)))
		# print(self.grid_winners.size/2)
		# print(self.no_of_boards)
		if sum(sum(self.grid_winners == 1)) > self.grid_winners.size/2:
			self.isEnd = True
			return 1
		elif sum(sum(self.grid_winners == -1)) > self.grid_winners.size/2:
			self.isEnd = True
			return -1
		else:
			return 0

	def reward(self , action , player):
		[x,  y , z] = [action//(2*(self.size - 1)) , (action % (2*(self.size - 1)))//2 , action % 2]
		reward = 0
		if z == 0:
			if x > 0 and self.board[x-1 , y, 0] and self.board[y, x-1, 1] and self.board[y + 1, x - 1, 1]:
				self.grid_winners[x-1 , y] = player
				self.no_of_boards += 1
				reward += 1

			if x < size - 1 and self.board[x+1 , y, 0] and self.board[y, x, 1] and self.board[y+1, x, 1]:
				self.grid_winners[x , y] = player
				self.no_of_boards += 1
				reward += 1
		elif z == 1:
			if x < size - 1 and self.board[y+1, x, 0] and self.board[y, x, 0] and self.board[x+1, y, 1]:
				self.grid_winners[y , x] = player
				self.no_of_boards += 1
				reward += 1
			if x > 0  and self.board[x-1 , y, 1] and self.board[y, x-1, 0] and self.board[y+1, x-1, 0]:
				self.grid_winners[y , x-1] = player
				self.no_of_boards += 1
				reward += 1
		return reward

	def play(self, rounds=100):
		for i in range(rounds):
			# if i % 1000 == 0:
			# 	print("Rounds {}".format(i))
			print("Rounds {}".format(i))
			while not self.isEnd:
				# Player 1
				reward = 1
				with torch.no_grad():
					tr = transforms.ToTensor()
					outputs = self.p1.model(tr(self.board))
				outputs_numpy = outputs.numpy()

				self.p1.next_state_Q_value = outputs_numpy

				while reward >= 1 and not self.isEnd:
					# print("---------------------------------------------------------")
					actions = self.available_actions()
					# print(actions)
					# self.p1.current_state_Q_value = self.p1.next_state_Q_value
					# print(max(self.p1.next_state_Q_value))
					p1_action = self.p1.chooseAction(actions, self.board)
					current_Q_value = self.p1.next_state_Q_value[p1_action]
					reward = self.reward(p1_action , self.p1.symbol)
					winner = self.winner()
					# if winner == 1 or winner == -1:
					# 	print(p1.symbol , winner)
					if winner == p1.symbol:
						# print("entered")
						reward += self.win_reward 
						self.winners.append(p1.symbol)
					if self.no_of_boards == self.total_squares:
						target = reward
						self.isEnd = True
					# take action and upate board state
					else:
						self.old_board = self.board.copy()
						self.updatestate(p1_action)
						with torch.no_grad():
							tr = transforms.ToTensor()
							outputs = self.p1.model(tr(self.board))
						outputs_numpy = outputs.numpy()
						self.p1.next_state_Q_value = outputs_numpy
						# print(outputs_numpy)
						# print(self.grid_winners)
						# print(self.available_actions())
						valid_outputs = outputs_numpy[self.available_actions()]
						Q_value = np.max(valid_outputs)
						target = reward + Q_value
					# if p1_action in [16, 34, 145, 163]:
					# print( reward , "p1")
					self.p1.optimizer.zero_grad()
					# Calculate loss and update
					with torch.set_grad_enabled(True):
						# target = torch.tensor(target).cpu().float()
						tr = transforms.ToTensor()
						current_Q_value = self.p1.model(tr(self.old_board))
						target_Q = torch.tensor(current_Q_value)
						target_Q[p1_action] = target
						# print(target_Q)
						loss = self.loss(current_Q_value , target_Q)
						loss.backward()
						self.p1.optimizer.step()

				reward = 1
				with torch.no_grad():
					tr = transforms.ToTensor()
					outputs = self.p2.model(tr(self.board))
				outputs_numpy = outputs.numpy()
				self.p2.next_state_Q_value = outputs_numpy
				while(reward >= 1 and not self.isEnd):
					# print("-------------------------------------------------------")
					actions = self.available_actions()
					# print(len(actions))
					# self.p1.current_state_Q_value = self.p1.next_state_Q_value
					p2_action = self.p2.chooseAction(actions, self.board)
					current_Q_value = self.p2.next_state_Q_value[p2_action]
					reward = self.reward(p2_action , self.p2.symbol)
					winner = self.winner()
					# if winner == 1 or winner == -1:
						# print(p2.symbol , winner)
					if winner == p2.symbol:
						self.winners.append(winner)
						# print("entered")
						reward += self.win_reward 
					if self.no_of_boards == self.total_squares:
						target = reward
						self.isEnd = True
					# take action and upate board state
					else:
						self.old_board = self.board.copy()
						self.updatestate(p2_action)
						with torch.no_grad():
							tr = transforms.ToTensor()
							outputs = self.p2.model(tr(self.board))
						outputs_numpy = outputs.numpy()
						self.p2.next_state_Q_value = outputs_numpy
						# print(len(self.available_actions))
						# print(self.grid_winners)
						# print(self.available_actions())
						valid_outputs = outputs_numpy[self.available_actions()]
						Q_value = np.max(valid_outputs)
						target = reward + Q_value
					# if p2_action in [16, 34, 145, 163]:
					# print(p2_action , reward , "p2")
					self.p2.optimizer.zero_grad()
					with torch.set_grad_enabled(True):
						# target = torch.tensor(target).cpu().float()
						tr = transforms.ToTensor()
						current_Q_value = self.p2.model(tr(self.old_board))
						target_Q = torch.tensor(current_Q_value)
						# print(target)
						target_Q[p2_action] = target
						loss = self.loss(current_Q_value.float() , target_Q.float())
						# print(loss)
						loss.backward()
						self.p2.optimizer.step()
					# for parameters in self.p2.model.parameters():
					# 	print(parameters.grad)
			# print(reward)
			self.p1.savePolicy()
			self.p2.savePolicy()
			self.p1.scheduler.step()
			self.p2.scheduler.step()
			# print(self.board)
			# print(self.grid_winners)
			self.reset()
			# print(self.p2.next_state_Q_value)

						
				# board_hash = self.getHash()
				# self.p1.addState(board_hash)
				# check board status if it is end

	def change_turn(self, X):
		if X == "A":
			return "B"
		if X == "B":
			return "A"

	def play2(self):
		scr = Screen(self.size)
		turn = "B"
		scr.show(self.board, self.grid_winners)
		self.p2.model = Model(self.size)
		self.p2.model.load_state_dict(torch.load("data/" + self.p2.name + "_Q_10_dict"))
		tr = transforms.ToTensor()
		# print(self.p2.model(tr(self.board)))
		# A is human
		while not self.isEnd:
			# print("self.isEnd" , self.isEnd)
			for event in pygame.event.get():
				if turn == "B":
					# quit the game when the player closes it
					reward = 0
					if event.type == pygame.QUIT:
						pygame.quit()
						exit(0)

					# left click
					elif event.type == pygame.MOUSEBUTTONDOWN and pygame.mouse.get_pressed()[0]:
						# if not self.accept_clicks:
						#     continue

						# get the current position of the cursor
						x = pygame.mouse.get_pos()[0]
						y = pygame.mouse.get_pos()[1]

						# check whether it was a not set wall that was clicked
						z = scr.get_wall(x, y, self.board)
						print("z : ", z)
						if z == -1:
							continue



					#  z to action number

						# print("---------------------------------------------------------")
						available_actions = self.available_actions()
						chosen_action = self.tuple_to_number(z)
						if chosen_action not in available_actions:
							continue
						# print(actions)
						# self.p2.current_state_Q_value = self.p2.next_state_Q_value
						reward = self.reward(chosen_action , self.p1.symbol)
						winner = self.winner()
						if winner == p1.symbol:
							reward += 10
							self.winners.append(winner)
						if self.no_of_boards == self.total_squares:
							target = reward
							self.isEnd = True
						# take action and upate board state
						else:
							self.old_board = self.board.copy()
							self.updatestate(chosen_action)
						scr.show(self.board, self.grid_winners)
						if reward >= 1:
							continue
						else :
							turn = self.change_turn(turn)


			if turn == "A":
				print("came here")			
				reward = 1
				with torch.no_grad():
					tr = transforms.ToTensor()
					outputs = self.p2.model(tr(self.board))
				outputs_numpy = outputs.numpy()
				self.p2.next_state_Q_value = outputs_numpy
				while(reward >= 1 and not self.isEnd):
					# print("-------------------------------------------------------")
					actions = self.available_actions()
					# print(len(actions))
					# self.p2.current_state_Q_value = self.p2.next_state_Q_value
					p2_action = self.p2.chooseAction(actions, self.board)
					print( self.p2.next_state_Q_value)
					# pdb.set_trace()
					current_Q_value = self.p2.next_state_Q_value[p2_action]
					reward = self.reward(p2_action , self.p2.symbol)
					winner = self.winner()
					if winner == p2.symbol:
						reward += 10
						self.winners.append(winner)
					if self.no_of_boards == self.total_squares:
						target = reward
						self.isEnd = True
					# take action and upate board state
					else:
						self.old_board = self.board.copy()
						self.updatestate(p2_action)
					scr.show(self.board, self.grid_winners)
				turn = self.change_turn(turn)




				
	
	# def available_positions
lr_ = 0.00001
exp_rate_ = 0.

class Player:
	def __init__(self, name, size, symbol, exp_rate=exp_rate_):
		self.name = name
		self.exp_rate = exp_rate
		# self.states = []  # record all positions taken
		# self.lr = 0.2
		# self.exp_rate = exp_rate
		# self.decay_gamma = 0.9
		self.model = Model(size).cpu()
		self.current_Q_value = 0
		self.symbol = symbol
		self.optimizer = optim.Adam(self.model.parameters(), lr=lr_)
		self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.2)
		self.counter = 0
	# def getHash(self, board):
	# 	boardHash = str(board.reshape(board.shape[0] * bpard.shape[1] * board.shape[2]))
	# 	return boardHash

	def chooseAction(self, actions, current_board):
		if np.random.uniform(0, 1) <= self.exp_rate:
			# take random action
			# idx = np.random.choice(len(positions))
			# print(actions)
			action = np.random.choice(actions)
			self.random_action_taken = True
		else:
			outputs = self.next_state_Q_value
			# outputs = outputs.numpy()
			# print(outputs.shape)
			# print(actions)
			valid_outputs = outputs[actions]
			action = actions[np.argmax(valid_outputs)]
			self.current_Q_value = np.max(valid_outputs)
			self.random_action_taken = False
		# print("{} takes action {}".format(self.name, action))
		self.counter += 1
		return action

	# append a hash state
	# def addState(self, state):
	# 	self.states.append(state)

	# # at the end of game, backpropagate and update states value
	# def feedReward(self, reward):
	# 	for st in reversed(self.states):
	# 		if self.states_value.get(st) is None:
	# 			self.states_value[st] = 0
	# 		self.states_value[st] += self.lr * (self.decay_gamma * reward - self.states_value[st])
	# 		reward = self.states_value[st]

	# def reset(self):
	# 	self.states = []

	def savePolicy(self):
		torch.save(self.model.state_dict() , 'data/' + self.name + "_Q_10_dict")

	# def loadPolicy(self, file):
	# 	fr = open(file, 'rb')
	# 	self.states_value = pickle.load(fr)
	# 	fr.close()


class Screen:
	def __init__(self, n=3):
		# initialize pygame
		pygame.init()
		self.gap = 40
		self.n = n
		# set the display size (one slot has 30x30 pixels; Walls: 4x26 Box: 26x26)
		self.screen = pygame.display.set_mode([n*30 + self.gap + 4, 30*n + self.gap + 4])

		# load all images
		self.empty = pygame.image.load("pics/empty.png")
		self.A = pygame.image.load("pics/A.png")
		self.B = pygame.image.load("pics/B.png")
		self.X = pygame.image.load("pics/X.png")
		self.block = pygame.image.load("pics/block.png")
		self.lineX = pygame.image.load("pics/lineX.png")
		self.lineXempty = pygame.image.load("pics/lineXempty.png")
		self.lineY = pygame.image.load("pics/lineY.png")
		self.lineYempty = pygame.image.load("pics/lineYempty.png")
		self.color = [155, 155, 155]
		self.screen.fill(self.color)
		pygame.display.update()
		# self.show()



	def show(self, p, wins):
		# print("P : \n", p)
		# print("wins : \n:", wins)
		for i in range(self.n):
			for j in range(self.n):
				x, y = i*30 + self.gap, j*30 + self.gap
				self.screen.blit(self.block, (x, y))
		for r in range(len(p)):
			for c in range(len(p[r])):

				k1 = p[r][c][0]
				x, y = c*30 + self.gap, r*30 + self.gap
				x += 4
				# y += 4
				if not k1:
					self.screen.blit(self.lineXempty, (x, y))
				else:
					self.screen.blit(self.lineX, (x, y))

				k2 = p[r][c][1]
				x, y = r*30 + self.gap, c*30 + self.gap
				y += 4
				if not k2:
					self.screen.blit(self.lineYempty, (x, y))
				else:
					self.screen.blit(self.lineY, (x, y))

		for r in range(len(wins)):
			for c in range(len(wins[r])):
				x, y = c*30 + self.gap, r*30 + self.gap
				x += 4
				y += 4
				k = wins[r][c]
				if k == 0:
					self.screen.blit(self.empty, (x, y))
				elif k == 1:
					self.screen.blit(self.A, (x, y))
				elif k == -1:
					self.screen.blit(self.B, (x, y))


		pygame.display.flip()

	def get_wall(self, x, y, p):
		x -= self.gap
		y -= self.gap
		row = x//30
		col = y//30
		if x%30 <= 4 : 
			# if p[col][row][1]:
			# 	return -1
			return (row, col, 1)
		elif y%30 <=4:
			# if p[row][col][0]:
			# 	return -1
			return (col, row, 0)
		return -1


if __name__ == "__main__":
	# training
	if int(sys.argv[2]):
		size = int(sys.argv[1])
		p1 = Player("p1" , size , 1)
		p2 = Player("p2" , size , -1)

		st = State(p1, p2 , size)
		print("training...")
		st.play(100)
		st.winners = np.array(st.winners)
		print(sum(st.winners == 1))
		print(sum(st.winners == -1))
		print(st.winners)

	else:
		# play with human
		# p1 = Player("computer", exp_rate=0)
		# p1.loadPolicy("policy_p1")

		# p2 = HumanPlayer("human")

		# st = State(p1, p2)
		# st.play2()
		# n = int(sys.argv[1])
		# scr = Screen(n)
		# p = np.random.randint(2, size=(n, n-1, 2))
		# wins = np.random.randint(3, size=(n-1, n-1))
		# wins -= 1
		# # p = np.load('p.npy')
		# # while True:
		# import time
		# scr.show(p, wins)
		# time.sleep(30)
		size = int(sys.argv[1])
		p1 = Player("p1" , size , 1 , 0)
		p2 = Player("p2" , size , -1 , 0)

		st = State(p1, p2 , size)
		print("training...")
		st.play2()
		st.winners = np.array(st.winners)
		print(sum(st.winners == 1))
		print(sum(st.winners == -1))
		print(st.winners)

